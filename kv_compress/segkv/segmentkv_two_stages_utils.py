import torch
import torch.nn as nn
import math


class TwoStageSegmentKV:
    def __init__(self, window_size=32, intermediate_capacity=4096, final_capacity=256, lookahead_steps=1, segments=None):
        self.intermediate_capacity = intermediate_capacity  # 第一阶段保留容量 (e.g., 4096)
        self.final_capacity = final_capacity  # 第二阶段最终容量 (e.g., 256)
        self.window_size = window_size

        assert segments is not None
        self.segments = segments

        # 内部状态记录
        self.is_compressed_stage1 = False
        self.is_compressed_stage2 = False
        self.kept_indices_stage1 = None  # 记录第一阶段留下的token在原始序列中的位置
        self.token2seg_stage1 = None  # 记录第一阶段留下的token分别属于哪个句子

        self.lookahead_steps = lookahead_steps
        self.gen_step = 0
        self.gen_query = None

        self.window_query = None

    @staticmethod
    def get_separator(model_name):
        if model_name == "mistral_v0.2":
            punc_qa = [28723, 28804, 28808, 842, 1550, 918, 609, 11505]  # for QA
            punc_code = [28745, 13]  # code, et. al.
        elif model_name == 'llama-3.1-8b-instruct':
            punc_qa = [0, 13, 26, 30, 198, 201, 271, 280, 317, 319, 382, 401, 464, 497, 570, 627, 629, 662, 696, 741,
                       758, 803, 881, 886, 947, 948, 949, 1025, 1038, 1125, 1131, 1177, 1210, 1237, 1240, 1432, 1861,
                       1865, 1875, 1967, 1975, 1980, 2055, 2195, 2266, 2268, 2412, 2564, 2595, 2622, 2885, 3001, 3147,
                       3155, 3238, 3304, 3317, 3343, 3382, 3618, 3677, 3707, 3840, 3945, 3961, 4095, 4098, 4286, 4390,
                       4489, 4527, 4682, 4710, 4713, 4772, 4926, 4965, 4999, 5051, 5139, 5146, 5233, 5240, 5244, 5322,
                       5354, 5378, 5380, 5585, 5638, 5904, 5984, 6030, 6053, 6087, 6088, 6188, 6242, 6266, 6466, 6470,
                       6552, 6905, 6911, 7112, 7352, 7377, 7470, 7673, 7722, 7801, 7849, 8054, 8295, 8361, 8731, 9135,
                       9456, 9522, 9602, 9772, 9801, 9938, 10246, 10359, 10380, 10927, 11131, 11261, 11371, 11390,
                       11690, 12106, 12241, 12275, 12340, 12515, 12795, 12872, 13352, 13507, 14421, 14670, 14697, 14719,
                       14781, 14963, 15276, 15328, 15364, 15399, 15425, 15552, 15656, 15725, 15759, 15804, 15850, 15882,
                       15908, 15993, 16016, 16049, 16123, 16291, 16571, 16656, 16715, 16853, 16971, 17386, 17523, 17556,
                       17642, 18072, 18086, 18304, 18575, 19103, 19449, 19691, 19789, 20837, 20838, 20979, 21366, 21537,
                       21671, 21732, 21908, 22174, 22307, 22414, 22431, 22438, 22445, 22666, 22669, 22877, 22896, 23137,
                       23169, 23253, 23341, 23632, 23811, 23849, 24287, 24314, 24482, 24688, 24984, 25332, 25430, 25433,
                       25638, 25750, 25765, 25782, 25833, 25863, 26104, 26525, 26543, 26570, 26575, 26637, 26722, 26727,
                       26815, 26977, 27001, 27074, 27118, 27135, 27218, 27261, 27333, 27676, 27677, 27708, 27880, 28105,
                       28212, 28288, 28452, 28527, 28633, 28684, 28871, 29001, 29175, 29216, 29249, 29275, 29307, 29773,
                       30222, 30251, 30284, 30629, 30662, 30736, 31118, 31134, 31318, 31538, 31558, 31931, 32091, 32318,
                       32325, 32339, 32395, 32407, 32483, 32807, 32897, 33157, 33414, 33674, 33696, 33698, 33736, 33970,
                       34115, 34193, 34794, 35016, 35047, 35277, 35432, 35683, 35749, 35929, 36284, 36397, 36434, 36776,
                       36796, 36886, 37049, 37094, 37274, 37280, 37307, 37428, 37434, 37468, 37752, 37918, 37925, 38011,
                       38028, 38489, 38503, 38526, 38545, 39502, 39709, 39839, 39876, 40125, 40465, 40567, 40673, 40778,
                       41107, 41400, 41417, 41418, 41430, 42229, 42395, 42448, 42755, 42943, 43058, 43112, 43369, 43484,
                       43492, 43550, 43584, 43888, 44384, 44497, 44607, 44684, 44708, 44969, 45026, 45198, 45222, 45751,
                       45991, 46114, 46196, 46200, 46228, 46449, 46726, 47191, 47656, 47839, 48014, 48122, 48165, 48366,
                       48469, 48556, 48586, 48663, 49089, 49420, 49440, 49711, 49827, 49998, 50274, 50522, 50655, 50954,
                       51064, 51447, 51767, 51905, 52130, 52417, 52607, 53172, 53368, 53394, 53670, 53699, 54605, 55128,
                       55160, 55236, 55303, 55591, 55638, 55797, 55816, 56535, 56631, 56822, 57019, 57057, 57341, 57604,
                       58173, 58490, 58670, 58877, 59162, 59171, 59257, 59437, 59475, 59601, 59824, 60001, 60057, 60582,
                       60857, 61560, 62073, 62121, 62361, 62420, 62450, 62499, 62757, 62927, 63963, 64023, 64065, 64273,
                       64654, 65066, 65353, 65429, 65981, 66534, 66689, 66820, 67090, 67471, 67476, 67606, 68005, 68229,
                       68393, 68551, 68664, 68896, 68964, 69113, 69134, 69201, 69738, 69846, 70455, 70593, 70594, 70717,
                       70746, 70900, 71038, 71090, 71131, 71291, 71928, 72066, 72246, 72509, 72571, 72576, 72734, 73186,
                       73203, 73433, 73489, 74031, 74438, 74578, 74630, 74763, 75000, 75100, 75208, 75223, 75303, 75485,
                       75546, 76658, 76683, 76794, 77540, 78405, 78455, 78760, 78867, 78928, 78941, 79033, 79310, 79354,
                       79798, 80183, 80239, 80241, 80246, 80301, 80318, 80395, 81028, 81031, 81344, 81430, 81734, 81903,
                       81923, 81974, 82084, 82191, 82472, 82492, 82508, 82968, 83056, 83106, 83316, 83461, 84021, 84243,
                       84670, 84909, 85445, 85676, 86418, 86640, 86645, 86695, 86715, 86992, 87172, 87297, 87314, 87332,
                       87527, 87879, 87953, 88136, 88673, 88686, 88785, 89030, 89612, 89883, 90014, 90046, 90520, 90820,
                       91292, 92614, 92653, 92912, 93204, 93313, 93343, 93479, 93955, 94068, 94093, 94145, 94147, 94489,
                       94497, 94733, 94770, 94996, 94997, 95022, 95110, 95152, 95181, 95241, 95434, 95621, 95899, 95992,
                       96395, 96477, 96592, 96770, 96774, 96915, 97186, 97821, 98195, 98356, 98533, 98656, 98668, 98722,
                       98929, 99264, 99419, 99501, 99573, 99627, 99858, 99888, 100064, 100073, 100158, 106688, 106838,
                       112532, 116833, 118599, 119197, 122604, 126408, 126437, 127729]

            # punc_qa = [0, 13, 30, 198, 201, 271, 319, 382, 497, 570, 627, 662, 696, 758, 881, 948, 949, 1038, 1131, 1210, 1240, 1432, 1861, 1865, 1875, 1975, 1980, 2055, 2195, 2266, 2268, 2564, 2595, 3001, 3238, 3304, 3343, 3677, 3707, 3961, 4095, 4286, 4390, 4489, 4527, 4682, 4710, 4713, 4999, 5139, 5240, 5244, 5354, 5380, 5585, 5638, 5904, 6053, 6087, 6188, 6242, 6266, 6905, 7352, 7377, 7673, 7801, 7849, 8054, 8731, 9135, 9456, 9522, 9602, 9772, 9801, 9938, 10246, 10380, 10927, 11261, 11371, 11690, 12106, 12241, 12340, 12515, 12795, 13352, 14670, 14697, 14781, 14963, 15364, 15425, 15552, 15725, 15804, 15850, 15882, 15993, 16049, 16656, 16715, 16853, 16971, 17386, 17523, 17642, 18072, 18304, 18575, 19103, 19449, 19691, 19789, 20837, 20838, 20979, 21537, 21908, 22414, 22438, 22666, 22877, 23253, 23811, 23849, 24287, 24314, 24482, 24688, 25332, 25430, 25638, 25750, 25765, 25782, 25833, 25863, 26104, 26543, 26575, 26722, 26815, 27001, 27074, 27135, 27218, 27261, 27333, 27676, 27708, 28212, 28452, 28527, 28684, 28871, 29001, 29175, 29249, 29275, 29773, 30251, 30284, 30629, 30662, 30736, 31118, 31134, 31318, 31538, 31931, 32091, 32318, 32325, 32483, 32807, 33157, 33674, 33698, 33970, 34115, 34193, 34794, 35016, 35047, 35277, 35432, 35683, 35929, 36284, 36397, 36434, 36776, 36886, 37049, 37280, 37307, 37434, 37752, 37918, 37925, 38011, 38028, 38526, 39709, 39839, 40125, 40465, 40567, 40673, 41417, 41418, 42229, 42395, 42448, 43369, 43492, 43550, 43584, 44607, 44684, 44708, 44969, 45198, 45751, 45991, 46114, 46196, 46200, 46228, 46449, 46726, 47656, 47839, 48014, 48122, 48165, 48366, 48469, 48556, 48586, 49089, 49420, 49711, 49827, 49998, 50522, 50655, 51064, 51447, 51767, 51905, 52130, 52417, 52607, 53172, 53368, 53394, 53670, 54605, 55128, 55160, 55638, 55797, 55816, 56822, 57057, 57341, 57604, 58490, 58670, 58877, 59162, 59171, 59475, 59601, 60001, 60582, 61560, 62073, 62361, 62450, 62499, 62757, 62927, 64023, 64065, 64273, 64654, 65066, 65981, 66820, 67090, 67476, 67606, 68229, 68393, 68664, 68964, 69113, 69134, 69201, 69738, 70455, 70900, 71090, 71131, 71291, 71928, 72066, 72734, 73203, 73489, 74438, 74578, 74630, 75000, 75100, 75208, 75223, 75303, 76683, 76794, 78405, 78867, 78928, 78941, 79310, 80183, 80239, 80241, 80301, 80395, 81028, 81031, 81344, 81430, 81734, 81923, 81974, 82084, 82472, 82492, 82508, 83106, 83461, 84243, 84909, 86645, 86715, 86992, 87172, 87314, 87527, 87879, 88136, 88673, 88686, 88785, 89030, 89612, 89883, 90014, 90046, 90520, 92614, 92653, 93204, 93479, 93955, 94068, 94093, 94770, 94996, 95022, 95110, 95181, 95241, 95434, 95621, 95899, 96395, 96477, 97186, 98356, 98656, 98722, 99501, 99627, 99858, 100073, 100158, 106688, 116833, 118599, 122604, 126437]


            punc_code = [25, 26, 198, 271, 280, 317, 340, 341, 401, 457, 487, 512, 517, 534, 551, 557, 629, 633, 696, 702, 720, 794, 803, 886, 933, 947, 997, 1025, 1038, 1084, 1125, 1158, 1177, 1192, 1232, 1237, 1270, 1287, 1329, 1432, 1465, 1473, 1504, 1657, 1680, 1763, 1827, 1875, 2355, 2368, 2412, 2499, 2595, 2608, 2622, 2652, 2670, 2820, 3033, 3086, 3147, 3270, 3317, 3382, 3456, 3559, 3588, 3602, 3707, 3718, 3730, 3818, 3840, 3945, 3961, 4098, 4352, 4485, 4489, 4532, 4555, 4649, 4713, 4764, 4772, 4815, 4926, 5051, 5146, 5233, 5235, 5240, 5243, 5244, 5322, 5378, 5537, 5555, 5572, 5680, 5731, 5787, 5984, 5996, 6030, 6054, 6087, 6360, 6394, 6466, 6470, 6494, 6694, 6911, 7048, 7071, 7112, 7132, 7171, 7275, 7361, 7468, 7519, 7700, 7775, 7887, 8044, 8256, 8295, 8361, 8555, 9000, 9122, 9505, 9586, 9658, 9763, 9801, 9946, 10162, 10294, 10359, 10556, 10661, 10792, 10912, 11147, 11187, 11200, 11290, 11308, 11414, 11444, 11498, 12038, 12578, 12586, 12713, 12795, 12872, 13090, 13251, 13320, 13385, 13503, 13507, 13549, 13906, 13981, 14053, 14211, 14419, 14440, 14501, 14557, 14623, 14719, 14790, 14838, 14852, 14963, 15018, 15029, 15053, 15073, 15152, 15162, 15276, 15328, 15399, 15425, 15497, 15786, 15908, 15993, 16049, 16052, 16244, 16484, 16487, 16554, 16823, 16919, 16925, 17122, 17311, 17398, 18026, 18086, 18309, 18605, 18888, 19014, 19086, 19124, 19500, 19548, 19652, 19691, 19789, 19800, 19938, 20084, 20495, 20667, 20923, 20979, 20999, 21366, 21671, 21711, 21916, 21932, 22414, 22428, 22431, 22445, 22623, 22669, 22844, 22896, 22953, 23094, 23113, 23137, 23341, 23535, 23547, 23584, 23631, 23632, 24287, 24333, 24356, 24371, 24546, 24984, 25312, 25393, 25433, 25638, 25863, 26027, 26315, 26510, 26543, 26582, 26652, 26706, 26722, 27001, 27164, 27355, 27381, 27482, 27644, 27677, 27788, 27829, 27907, 28225, 28243, 28411, 28416, 28452, 28581, 28633, 28871, 28925, 29138, 29175, 29216, 29307, 29347, 29361, 29436, 29448, 29494, 29812, 29896, 30222, 30424, 30936, 31411, 31879, 32049, 32339, 32395, 32407, 32583, 32815, 32897, 33006, 33696, 33736, 33968, 34113, 34358, 34451, 34726, 34741, 34766, 34794, 34834, 34962, 35033, 35049, 35235, 35683, 35742, 36039, 36217, 36348, 36411, 36474, 36566, 36929, 36933, 36955, 37428, 37677, 37815, 37878, 37903, 37945, 38028, 38151, 38489, 38890, 39545, 39597, 39876, 40667, 40965, 40987, 41400, 41430, 41437, 42064, 42265, 42720, 42736, 42793, 42943, 43058, 43232, 43373, 43484, 43550, 44160, 44253, 44370, 44384, 44441, 44497, 44520, 44708, 45026, 45294, 45416, 45751, 45765, 45832, 45835, 46086, 46200, 46228, 46420, 46907, 46930, 47082, 47251, 47839, 47973, 48436, 48546, 48549, 48663, 49015, 49089, 49215, 49440, 49526, 49671, 49722, 50677, 50724, 50954, 51087, 51672, 51780, 51860, 52070, 52224, 52463, 52518, 53266, 53340, 53394, 53472, 53562, 53581, 53699, 53820, 54605, 54732, 54840, 55095, 55160, 55802, 56244, 56366, 56521, 56530, 56547, 57019, 57173, 57696, 57879, 57931, 58093, 58150, 58230, 58420, 58451, 58670, 58821, 59056, 59171, 59257, 59277, 59301, 59510, 59518, 60058, 60201, 60309, 60503, 60609, 60749, 60857, 60892, 60945, 61028, 61340, 61659, 62098, 62377, 62539, 63356, 63449, 63963, 63987, 64140, 64259, 64577, 64736, 65066, 65239, 66152, 66768, 66915, 67471, 67476, 68005, 68166, 68414, 68536, 68725, 68964, 69201, 69265, 69662, 70466, 70594, 70650, 70977, 71254, 71280, 72075, 72246, 72330, 72348, 72462, 72509, 72571, 72572, 72712, 72879, 73203, 73489, 74031, 74131, 74463, 74763, 74922, 75064, 75484, 75485, 75546, 75591, 75625, 76126, 76153, 76328, 76452, 76567, 77010, 77425, 77559, 77868, 78661, 78760, 78887, 79033, 79055, 79078, 79093, 79237, 79414, 79772, 80183, 80233, 80318, 80326, 80839, 81031, 81605, 81787, 81819, 82191, 82261, 82745, 82867, 82968, 82992, 83056, 83316, 83461, 83793, 83993, 84021, 84107, 84420, 84547, 84585, 84763, 84909, 84953, 85013, 85312, 85676, 85736, 86029, 86590, 86704, 86717, 87025, 87297, 87421, 87809, 87870, 87927, 87953, 88137, 88179, 88241, 88686, 88728, 88776, 88994, 89580, 89904, 89953, 90046, 90098, 90353, 90362, 91322, 91406, 91508, 91788, 92012, 92615, 92681, 93035, 93110, 93263, 93449, 93823, 93853, 93863, 94104, 94145, 94296, 94344, 94696, 94733, 94854, 94973, 94997, 95022, 95241, 95339, 95435, 95988, 96047, 96163, 96423, 96477, 96592, 96742, 96915, 97117, 97300, 97435, 97821, 98195, 98705, 98852, 98929, 99179, 99374, 99522, 99888, 100107, 101060, 101212, 101986, 102795, 103403, 104318, 107360, 109850, 109872, 109926, 110819, 112868, 115027, 116914, 118097, 127729]

        else:
            raise NotImplementedError
        return punc_qa, punc_code

    @staticmethod
    def get_segments(input_ids, model_name, window_size=32):
        bs, seq_len = input_ids.shape
        assert bs == 1
        # max_len = seq_len
        max_len = seq_len - window_size
        device = input_ids.device

        punc_qa, punc_code = TwoStageSegmentKV.get_separator(model_name)

        punc_qa = torch.tensor(punc_qa, device=device, dtype=input_ids.dtype)
        mask_qa = torch.isin(input_ids[0], punc_qa)
        punc_position_qa = torch.nonzero(mask_qa, as_tuple=False).reshape(-1)

        punc_code = torch.tensor(punc_code, device=device, dtype=input_ids.dtype)
        mask_code = torch.isin(input_ids[0], punc_code)
        punc_position_code = torch.nonzero(mask_code, as_tuple=False).reshape(-1)

        punc_position = punc_position_qa if punc_position_qa.sum() > punc_position_code.sum() else punc_position_code

        bounds = punc_position[punc_position < max_len] + 1  # +1 causes the punctuation mark to be placed in its corresponding paragraph.
        bounds = bounds.sort(descending=False).values

        start = torch.tensor([0], device=device)
        end = torch.tensor([max_len], device=device)
        if end != bounds[-1]:
            segments = torch.cat([start, bounds, end]).to(device)
        else:
            segments = torch.cat([start, bounds]).to(device)

        return segments

    def _get_token2seg(self, num_tokens, bsz, num_heads):
        """根据segments划分，计算原始序列中每个token对应的句子索引"""
        device = self.segments.device
        positions = torch.arange(num_tokens, device=device)
        # right=True 确保标点属于前一句
        token2seg = torch.searchsorted(self.segments, positions, right=True) - 1
        token2seg = token2seg.unsqueeze(0).expand(bsz, num_heads, -1)
        return token2seg

    def calcul_scores_with_query(self, query_states, key_states, token2seg_map, num_segments):
        """通用分数计算：给定Query和Key，计算句子级和Token级分数"""
        bsz, num_heads, q_len, head_dim = query_states.shape
        device = query_states.device

        # 1. 计算 Attention Weights (QK^T)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        # 如果是Prefill阶段自注意，需要因果掩码；如果是Decoding阶段对之前的KV，则不需要
        # if q_len > 1:
        mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -q_len:, -q_len:] += mask[None, None, :, :]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 2. 聚合为token 分数 (Mean over observation window)
        # [bsz, num_heads, k_len]
        token_important = attn_weights[:, :, -q_len:, :-q_len].mean(dim=-2)

        # 使用多个gen_token校正
        if self.gen_step == self.lookahead_steps:
            # token_important = token_important[:, :, :-self.window_size]
            # token_important = token_important[:, :, :-(self.window_size-4)]
            token_important = token_important

        # 3. 按句子索引累加
        indices = token2seg_map
        seg_scores = torch.zeros((bsz, num_heads, num_segments), device=device, dtype=attn_weights.dtype)
        seg_scores = seg_scores.scatter_add(dim=-1, index=indices, src=token_important)

        # 4. 除以句长获得平均分 (由于Stage 1可能删除了部分token，这里推荐使用原始句长或当前保留句长)
        # 这里为了简单，直接计算当前存在的token对应的句长
        ones = torch.ones_like(token_important)
        seg_counts = torch.zeros((bsz, num_heads, num_segments), device=device, dtype=attn_weights.dtype)
        seg_counts = seg_counts.scatter_add(dim=-1, index=indices, src=ones)
        seg_scores = seg_scores / (seg_counts + 1e-6)

        # 5. 回传给Token
        token_scores = seg_scores.gather(dim=-1, index=indices)
        return token_scores

    def update_kv(self, query_states, key_states, value_states):
        bsz, num_heads, seq_len, head_dim = query_states.shape
        num_segments = self.segments.shape[0] - 1

        # --- 长度尚短，不压缩 ---
        if seq_len < self.intermediate_capacity and not self.is_compressed_stage1:
            
            self.window_query = query_states[:, :, -32:, :]

            self.kept_indices_stage1 = torch.arange(seq_len - self.window_size, device=query_states.device).expand(bsz, num_heads, -1)
            self.token2seg_stage1 = self._get_token2seg(seq_len - self.window_size, bsz, num_heads)

            self.is_compressed_stage1 = True
            return key_states, value_states

        # --- 情况 1: 第一阶段压缩 (Prefill 结束时) ---
        # 判定条件：当前是Prefill（q_len > 1）且尚未进行一阶段压缩
        if seq_len > 1 and not self.is_compressed_stage1:

            self.window_query = query_states[:, :, -32:, :]

            token2seg_full = self._get_token2seg(seq_len - self.window_size, bsz, num_heads)

            # 使用最后 window_size 个 token 作为观察窗口计算分数
            # 注意：这里计算的是前 seq_len - window_size 个 token 的重要性
            token_scores = self.calcul_scores_with_query(
                query_states[:, :, -self.window_size:, :],
                key_states,
                token2seg_full,
                num_segments
            )

            # 挑选 Top-K
            indices = token_scores.topk(self.intermediate_capacity - self.window_size, dim=-1).indices

            # 记录保留的信息用于第二阶段
            self.kept_indices_stage1 = indices
            self.token2seg_stage1 = token2seg_full.gather(dim=-1, index=indices)

            # 执行压缩
            k_past = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            v_past = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

            self.is_compressed_stage1 = True
            return torch.cat([k_past, key_states[:, :, -self.window_size:, :]], dim=2), \
                torch.cat([v_past, value_states[:, :, -self.window_size:, :]], dim=2)

        # --- 情况 2: 第二阶段压缩 (生成第一个 Token 时) ---
        # 当前是Decoding（q_len == 1），且已完成一阶段但未完成二阶段，且小于第二次压缩budget。不压缩，直接返回
        current_key_len = key_states.shape[2]
        if seq_len == 1 and self.is_compressed_stage1 and not self.is_compressed_stage2 and current_key_len < self.final_capacity:
            
            self.is_compressed_stage2 = True
            # 后续为了节省内存，可以清空中间变量
            self.token2seg_stage1 = None
            return key_states, value_states


        # 当前是Decoding（q_len == 1），且已完成一阶段但未完成二阶段
        if seq_len == 1 and self.is_compressed_stage1 and not self.is_compressed_stage2:
            self.gen_step = self.gen_step + 1
            # print(self.gen_step)
            # print(self.lookahead_steps)

            if self.gen_query is None:
                self.gen_query = query_states
            else:
                self.gen_query = torch.cat([self.gen_query, query_states], dim=-2)

            if self.gen_step == self.lookahead_steps:

                cur_window_size = self.window_size + self.gen_step

                self.gen_query = torch.cat([self.window_query, self.gen_query], dim=-2) 

                token_scores = self.calcul_scores_with_query(
                    self.gen_query,
                    key_states,
                    self.token2seg_stage1,
                    num_segments
                )

                indices = token_scores.topk(self.final_capacity - cur_window_size, dim=-1).indices

                # 执行最终压缩
                k_final_past = key_states[:, :, :-cur_window_size, :].gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                v_final_past = value_states[:, :, :-cur_window_size, :].gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

                self.is_compressed_stage2 = True
                # 后续为了节省内存，可以清空中间变量
                self.token2seg_stage1 = None
                self.gen_query = None
                self.window_query = None

                return torch.cat([k_final_past, key_states[:, :, -cur_window_size:, :]], dim=2), torch.cat([v_final_past, value_states[:, :, -cur_window_size:, :]], dim=2)
            else:
                return key_states, value_states

        # --- 情况 3: 已经完成两阶段压缩，后续解码阶段正常拼接 ---
        return key_states, value_states


def init_twostage_segmentkv(self):
    if not hasattr(self, "kv_comp"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'intermediate_capacity'):
            self.config.intermediate_capacity = 2048
        if not hasattr(self.config, 'max_capacity'):
            self.config.max_capacity = 256
        if not hasattr(self.config, 'lookahead_steps'):
            self.config.lookahead_steps = 4
        assert hasattr(self.config, 'segments')

    self.kv_comp = TwoStageSegmentKV(
        window_size = self.config.window_size,
        intermediate_capacity = self.config.intermediate_capacity,
        final_capacity = self.config.max_capacity,
        lookahead_steps=self.config.lookahead_steps,
        segments=self.config.segments,
        )