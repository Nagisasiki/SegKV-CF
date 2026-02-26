import torch
from transformers.cache_utils import Cache
from typing import List, Iterable, Dict, Any, Optional, Tuple


class DynamicCache_LAQ(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, _distributed_cache_data=None, num_road: int = 4):
        super().__init__(layer_class_to_replicate=type(self))

        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.temp_local_window_query: List[torch.Tensor] = []
        self.num_road = num_road

        if _distributed_cache_data is not None:
            for key_states, value_states in _distributed_cache_data:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
            self,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            window_size: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            # print(key_states.shape)
            # key_states = key_states.mean(dim=0, keepdim=True)
            # key_states = key_states.repeat(self.num_road, 1, 1, 1)
            # value_states = value_states.mean(dim=0, keepdim=True)
            # value_states = value_states.repeat(self.num_road, 1, 1, 1)
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.temp_local_window_query.append(query_states[:, :, -window_size:, :])
                # print(window_size)
                # print(query_states[:,:,-window_size:,:].shape)
                # input()
            elif (
                    not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                self.temp_local_window_query[layer_idx] = torch.cat(
                    [self.temp_local_window_query[layer_idx], query_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
                len(self.key_cache) == 0  # no cache in any layer
                or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
                or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int) -> int:
        """
        Return how many cached KV tokens can be used for attention.
        For LAQ, this is simply the current compressed KV length.
        """
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    def choose_last_kv(self, index):
        for i in range(len(self.key_cache)):
            for num in range(self.num_road):
                self.key_cache[i][num, :, -1, :] = self.key_cache[i][index, :, -1, :]
                self.value_cache[i][num, :, -1, :] = self.value_cache[i][index, :, -1, :]

    def merge_last_kv(self, weigeht):
        for i in range(len(self.key_cache)):
            merge_key = (self.key_cache[i][:, :, -1, :] * weigeht.view(-1, 1, 1)).sum(dim=0)
            merge_value = (self.value_cache[i][:, :, -1, :] * weigeht.view(-1, 1, 1)).sum(dim=0)
            # print(merge_key.shape)
            # input()
            for num in range(self.num_road):
                # merge_key =
                self.key_cache[i][num, :, -1, :] = merge_key
                self.value_cache[i][num, :, -1, :] = merge_value

    def cover_last_kv(self, k, v, layer_idx):
        self.key_cache[layer_idx][:, :, -1, :] = k
        self.value_cache[layer_idx][:, :, -1, :] = v