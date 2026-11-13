import torch

class LinearStateUnit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = torch.nn.Parameter(torch.empty(1, 1, dtype=torch.float64), requires_grad=True)
        self.p1 = torch.nn.Parameter(torch.empty(1, 2, dtype=torch.float64), requires_grad=True)
        self.p2 = torch.nn.Parameter(torch.empty(1, 2, dtype=torch.float64), requires_grad=True)
        self.p3 = torch.nn.Parameter(torch.empty(1, 1, 2, 2, dtype=torch.float64), requires_grad=True)
        self.p4 = torch.nn.Parameter(torch.empty(1, 1, 2, 2, dtype=torch.float64), requires_grad=True)
        self.p5 = torch.nn.Parameter(torch.empty(1, 1, 2, 2, dtype=torch.float64), requires_grad=True)
        self.p6 = torch.nn.Parameter(torch.empty(1, 1, dtype=torch.float64), requires_grad=True)

    def _prepare(self):
        # f0000: constant 1.0
        c1 = torch.tensor(1.0, dtype=torch.float64)
        # f0001: constant 2.0
        c2 = torch.tensor(2.0, dtype=torch.float64)
        
        # f0002: exp(p0)
        f0002 = torch.exp(self.p0)
        # f0003: exp(p1)
        f0003 = torch.exp(self.p1)
        # f0004: neg(f0003)
        f0004 = torch.neg(f0003)
        # f0005: neg(p2)
        f0005 = torch.neg(self.p2)
        
        # f0006-f0008: extract p5[0,0,:,0] -> [1,2] -> [2]
        f0006 = self.p5[:, :, :, 0:1]
        f0007 = torch.squeeze(f0006, dim=3)
        f0008 = torch.squeeze(f0007, dim=0)
        
        # f0009-f0011: extract p5[0,0,:,1] -> [1,2] -> [2]
        f0009 = self.p5[:, :, :, 1:2]
        f0010 = torch.squeeze(f0009, dim=3)
        f0011 = torch.squeeze(f0010, dim=0)
        
        # f0012: neg(f0011)
        f0012 = torch.neg(f0011)
        
        # f0013-f0015: extract p3[0,0,:,0] -> [1,2] -> [2]
        f0013 = self.p3[:, :, :, 0:1]
        f0014 = torch.squeeze(f0013, dim=3)
        f0015 = torch.squeeze(f0014, dim=0)
        
        # f0016-f0018: extract p3[0,0,:,1] -> [1,2] -> [2]
        f0016 = self.p3[:, :, :, 1:2]
        f0017 = torch.squeeze(f0016, dim=3)
        f0018 = torch.squeeze(f0017, dim=0)
        
        # f0019-f0021: extract p4[0,0,:,0] -> [1,2] -> [2]
        f0019 = self.p4[:, :, :, 0:1]
        f0020 = torch.squeeze(f0019, dim=3)
        f0021 = torch.squeeze(f0020, dim=0)
        
        # f0022-f0024: extract p4[0,0,:,1] -> [1,2] -> [2]
        f0022 = self.p4[:, :, :, 1:2]
        f0023 = torch.squeeze(f0022, dim=3)
        f0024 = torch.squeeze(f0023, dim=0)
        
        # f0025: div(c2, f0002)
        f0025 = torch.div(c2, f0002)
        # f0026: sub(f0025, f0004)
        f0026 = torch.sub(f0025, f0004)
        # f0027: neg(f0005)
        f0027 = torch.neg(f0005)
        # f0028: mul(f0026, f0026)
        f0028 = torch.mul(f0026, f0026)
        # f0029: mul(f0027, f0027)
        f0029 = torch.mul(f0027, f0027)
        # f0030: add(f0028, f0029)
        f0030 = torch.add(f0028, f0029)
        # f0031: div(f0026, f0030)
        f0031 = torch.div(f0026, f0030)
        # f0032: neg(f0027)
        f0032 = torch.neg(f0027)
        # f0033: div(f0032, f0030)
        f0033 = torch.div(f0032, f0030)
        # f0034: add(f0025, f0004)
        f0034 = torch.add(f0025, f0004)
        # f0035: mul(f0008, f0031)
        f0035 = torch.mul(f0008, f0031)
        # f0036: mul(f0012, f0033)
        f0036 = torch.mul(f0012, f0033)
        # f0037: sub(f0035, f0036)
        f0037 = torch.sub(f0035, f0036)
        # f0038: mul(f0008, f0033)
        f0038 = torch.mul(f0008, f0033)
        # f0039: mul(f0012, f0031)
        f0039 = torch.mul(f0012, f0031)
        # f0040: add(f0038, f0039)
        f0040 = torch.add(f0038, f0039)
        # f0041: mul(f0037, f0008)
        f0041 = torch.mul(f0037, f0008)
        # f0042: mul(f0040, f0011)
        f0042 = torch.mul(f0040, f0011)
        # f0043: sub(f0041, f0042)
        f0043 = torch.sub(f0041, f0042)
        # f0047: sum(f0043, dim=1, keepdim=True)
        f0047 = torch.sum(f0043, dim=1, keepdim=True)
        # f0048: mul(c2, f0047)
        f0048 = torch.mul(c2, f0047)
        # f0049: add(c1, f0048)
        f0049 = torch.add(c1, f0048)
        # f0050: div(f0037, f0049)
        f0050 = torch.div(f0037, f0049)
        # f0051: div(f0040, f0049)
        f0051 = torch.div(f0040, f0049)
        
        return {
            'dr': f0031, 'di': f0033, 'er': f0034, 'ei': f0005,
            'pr': f0008, 'pi': f0011, 'qr': f0008, 'qi': f0012,
            'rr': f0050, 'ri': f0051, 'br': f0015, 'bi': f0018,
            'cr': f0021, 'ci': f0024
        }

    def _sequence_body(self, x, s, params):
        c2 = torch.tensor(2.0, dtype=torch.float64)
        
        # f0053-f0054: s[:, :, :, 0:1] -> squeeze(3) -> [B, 1, 2]
        f0053 = s[:, :, :, 0:1]
        f0054 = torch.squeeze(f0053, dim=3)
        
        # f0055-f0056: s[:, :, :, 1:2] -> squeeze(3) -> [B, 1, 2]
        f0055 = s[:, :, :, 1:2]
        f0056 = torch.squeeze(f0055, dim=3)
        
        # f0057: mul(qr, f0054)
        f0057 = torch.mul(params['qr'], f0054)
        # f0058: mul(qi, f0056)
        f0058 = torch.mul(params['qi'], f0056)
        # f0059: sub(f0057, f0058)
        f0059 = torch.sub(f0057, f0058)
        # f0063: sum(f0059, dim=2, keepdim=True) -> [B, 1, 1]
        f0063 = torch.sum(f0059, dim=2, keepdim=True)
        # f0064: mul(f0063, c2)
        f0064 = torch.mul(f0063, c2)
        
        # f0065: mul(er, f0054)
        f0065 = torch.mul(params['er'], f0054)
        # f0066: mul(ei, f0056)
        f0066 = torch.mul(params['ei'], f0056)
        # f0067: sub(f0065, f0066)
        f0067 = torch.sub(f0065, f0066)
        
        # f0068: mul(er, f0056)
        f0068 = torch.mul(params['er'], f0056)
        # f0069: mul(ei, f0054)
        f0069 = torch.mul(params['ei'], f0054)
        # f0070: add(f0068, f0069)
        f0070 = torch.add(f0068, f0069)
        
        # f0071: mul(pr, f0064)
        f0071 = torch.mul(params['pr'], f0064)
        # f0072: mul(pi, f0064)
        f0072 = torch.mul(params['pi'], f0064)
        
        # f0073: sub(f0067, f0071)
        f0073 = torch.sub(f0067, f0071)
        # f0074: sub(f0070, f0072)
        f0074 = torch.sub(f0070, f0072)
        
        # f0075: unsqueeze(x, dim=2) -> [B, 1, 1]
        f0075 = torch.unsqueeze(x, dim=2)
        # f0076: mul(c2, f0075)
        f0076 = torch.mul(c2, f0075)
        # f0077: mul(br, f0076)
        f0077 = torch.mul(params['br'], f0076)
        # f0078: mul(bi, f0076)
        f0078 = torch.mul(params['bi'], f0076)
        
        # f0079: add(f0073, f0077)
        f0079 = torch.add(f0073, f0077)
        # f0080: add(f0074, f0078)
        f0080 = torch.add(f0074, f0078)
        
        # f0081: mul(rr, f0079)
        f0081 = torch.mul(params['rr'], f0079)
        # f0082: mul(ri, f0080)
        f0082 = torch.mul(params['ri'], f0080)
        # f0083: sub(f0081, f0082)
        f0083 = torch.sub(f0081, f0082)
        # f0087: sum(f0083, dim=2, keepdim=True) -> [B, 1, 1]
        f0087 = torch.sum(f0083, dim=2, keepdim=True)
        # f0088: mul(f0087, c2)
        f0088 = torch.mul(f0087, c2)
        
        # f0089: mul(pr, f0088)
        f0089 = torch.mul(params['pr'], f0088)
        # f0090: mul(pi, f0088)
        f0090 = torch.mul(params['pi'], f0088)
        
        # f0091: sub(f0079, f0089)
        f0091 = torch.sub(f0079, f0089)
        # f0092: sub(f0080, f0090)
        f0092 = torch.sub(f0080, f0090)
        
        # f0093: mul(dr, f0091)
        f0093 = torch.mul(params['dr'], f0091)
        # f0094: mul(di, f0092)
        f0094 = torch.mul(params['di'], f0092)
        # f0095: sub(f0093, f0094)
        f0095 = torch.sub(f0093, f0094)
        
        # f0096: mul(dr, f0092)
        f0096 = torch.mul(params['dr'], f0092)
        # f0097: mul(di, f0091)
        f0097 = torch.mul(params['di'], f0091)
        # f0098: add(f0096, f0097)
        f0098 = torch.add(f0096, f0097)
        
        # f0099: stack(f0095, f0098, dim=3) -> [B, 1, 2, 2]
        s_next = torch.stack([f0095, f0098], dim=3)
        
        # f0100: mul(cr, f0095)
        f0100 = torch.mul(params['cr'], f0095)
        # f0101: mul(ci, f0098)
        f0101 = torch.mul(params['ci'], f0098)
        # f0102: sub(f0100, f0101)
        f0102 = torch.sub(f0100, f0101)
        # f0106: sum(f0102, dim=2, keepdim=False) -> [B, 1]
        f0106 = torch.sum(f0102, dim=2, keepdim=False)
        # f0107: mul(c2, f0106)
        v = torch.mul(c2, f0106)
        
        return v, s_next

    def _kernel_body(self, x, s, flag, params):
        c2 = torch.tensor(2.0, dtype=torch.float64)
        
        # f0109: zeros_like(x) -> [1, 1]
        f0109 = torch.zeros_like(x)
        # f0110: add(f0109, flag) -> [1, 1]
        f0110 = torch.add(f0109, flag)
        # f0111: zeros_like(flag) -> [1, 1]
        flag_next = torch.zeros_like(flag)
        
        # f0112-f0113: s[:, :, :, 0:1] -> squeeze(3) -> [1, 1, 2]
        f0112 = s[:, :, :, 0:1]
        f0113 = torch.squeeze(f0112, dim=3)
        
        # f0114-f0115: s[:, :, :, 1:2] -> squeeze(3) -> [1, 1, 2]
        f0114 = s[:, :, :, 1:2]
        f0115 = torch.squeeze(f0114, dim=3)
        
        # f0116: mul(qr, f0113)
        f0116 = torch.mul(params['qr'], f0113)
        # f0117: mul(qi, f0115)
        f0117 = torch.mul(params['qi'], f0115)
        # f0118: sub(f0116, f0117)
        f0118 = torch.sub(f0116, f0117)
        # f0122: sum(f0118, dim=2, keepdim=True) -> [1, 1, 1]
        f0122 = torch.sum(f0118, dim=2, keepdim=True)
        # f0123: mul(f0122, c2)
        f0123 = torch.mul(f0122, c2)
        
        # f0124: mul(er, f0113)
        f0124 = torch.mul(params['er'], f0113)
        # f0125: mul(ei, f0115)
        f0125 = torch.mul(params['ei'], f0115)
        # f0126: sub(f0124, f0125)
        f0126 = torch.sub(f0124, f0125)
        
        # f0127: mul(er, f0115)
        f0127 = torch.mul(params['er'], f0115)
        # f0128: mul(ei, f0113)
        f0128 = torch.mul(params['ei'], f0113)
        # f0129: add(f0127, f0128)
        f0129 = torch.add(f0127, f0128)
        
        # f0130: mul(pr, f0123)
        f0130 = torch.mul(params['pr'], f0123)
        # f0131: mul(pi, f0123)
        f0131 = torch.mul(params['pi'], f0123)
        
        # f0132: sub(f0126, f0130)
        f0132 = torch.sub(f0126, f0130)
        # f0133: sub(f0129, f0131)
        f0133 = torch.sub(f0129, f0131)
        
        # f0134: unsqueeze(f0110, dim=2) -> [1, 1, 1]
        f0134 = torch.unsqueeze(f0110, dim=2)
        # f0135: mul(c2, f0134)
        f0135 = torch.mul(c2, f0134)
        # f0136: mul(br, f0135)
        f0136 = torch.mul(params['br'], f0135)
        # f0137: mul(bi, f0135)
        f0137 = torch.mul(params['bi'], f0135)
        
        # f0138: add(f0132, f0136)
        f0138 = torch.add(f0132, f0136)
        # f0139: add(f0133, f0137)
        f0139 = torch.add(f0133, f0137)
        
        # f0140: mul(rr, f0138)
        f0140 = torch.mul(params['rr'], f0138)
        # f0141: mul(ri, f0139)
        f0141 = torch.mul(params['ri'], f0139)
        # f0142: sub(f0140, f0141)
        f0142 = torch.sub(f0140, f0141)
        # f0146: sum(f0142, dim=2, keepdim=True) -> [1, 1, 1]
        f0146 = torch.sum(f0142, dim=2, keepdim=True)
        # f0147: mul(f0146, c2)
        f0147 = torch.mul(f0146, c2)
        
        # f0148: mul(pr, f0147)
        f0148 = torch.mul(params['pr'], f0147)
        # f0149: mul(pi, f0147)
        f0149 = torch.mul(params['pi'], f0147)
        
        # f0150: sub(f0138, f0148)
        f0150 = torch.sub(f0138, f0148)
        # f0151: sub(f0139, f0149)
        f0151 = torch.sub(f0139, f0149)
        
        # f0152: mul(dr, f0150)
        f0152 = torch.mul(params['dr'], f0150)
        # f0153: mul(di, f0151)
        f0153 = torch.mul(params['di'], f0151)
        # f0154: sub(f0152, f0153)
        f0154 = torch.sub(f0152, f0153)
        
        # f0155: mul(dr, f0151)
        f0155 = torch.mul(params['dr'], f0151)
        # f0156: mul(di, f0150)
        f0156 = torch.mul(params['di'], f0150)
        # f0157: add(f0155, f0156)
        f0157 = torch.add(f0155, f0156)
        
        # f0158: stack(f0154, f0157, dim=3) -> [1, 1, 2, 2]
        s_next = torch.stack([f0154, f0157], dim=3)
        
        # f0159: mul(cr, f0154)
        f0159 = torch.mul(params['cr'], f0154)
        # f0160: mul(ci, f0157)
        f0160 = torch.mul(params['ci'], f0157)
        # f0161: sub(f0159, f0160)
        f0161 = torch.sub(f0159, f0160)
        # f0165: sum(f0161, dim=2, keepdim=False) -> [1, 1]
        f0165 = torch.sum(f0161, dim=2, keepdim=False)
        # f0166: mul(c2, f0165)
        v = torch.mul(c2, f0165)
        
        return v, s_next, flag_next

    def forward(self, u, h0):
        params = self._prepare()
        
        # sequence_scan
        T = u.shape[2]
        seq_outputs = []
        s = h0
        for t in range(T):
            x_t = u[:, :, t]  # [B, 1]
            v_t, s = self._sequence_body(x_t, s, params)
            seq_outputs.append(v_t)
        sequence = torch.stack(seq_outputs, dim=2) if T > 0 else torch.empty(u.shape[0], 1, 0, dtype=torch.float64)
        h_final = s
        
        # kernel_scan
        u_kernel = u[:, 0:1, :]  # [B, 1, T] -> slice axis 0, start 0, stop 1
        s_kernel = torch.zeros_like(self.p5)  # [1, 1, 2, 2]
        flag = torch.tensor([[1.0]], dtype=torch.float64)
        
        kernel_outputs = []
        for t in range(T):
            x_t = u_kernel[:, :, t]  # [B, 1] -> but kernel_body expects [1, 1] based on interface id006
            # Wait, kernel_scan input u is *id004 which is [1, 1, T]. f0167 slices unit.u [B,1,T] to [1,1,T].
            # So x_t is [1, 1].
            v_t, s_kernel, flag = self._kernel_body(x_t, s_kernel, flag, params)
            kernel_outputs.append(v_t)
        kernel = torch.stack(kernel_outputs, dim=2) if T > 0 else torch.empty(1, 1, 0, dtype=torch.float64)
        
        # y calculation
        p6_unsqueezed = torch.unsqueeze(self.p6, dim=2)  # [1, 1, 1]
        u_scaled = torch.mul(u, p6_unsqueezed)  # [B, 1, T]
        y = torch.add(sequence, u_scaled)  # [B, 1, T]
        
        return kernel, y, h_final


def create_model():
    return LinearStateUnit()