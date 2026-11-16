import torch
import torch.nn as nn
import torch.nn.functional as F

def build(values):
    class GatedStateBlock(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.q_weight = nn.Parameter(values['q_weight'].clone())
            self.q_bias = nn.Parameter(values['q_bias'].clone())
            self.k_weight = nn.Parameter(values['k_weight'].clone())
            self.k_bias = nn.Parameter(values['k_bias'].clone())
            self.v_weight = nn.Parameter(values['v_weight'].clone())
            self.v_bias = nn.Parameter(values['v_bias'].clone())
            self.out_weight = nn.Parameter(values['out_weight'].clone())
            self.out_bias = nn.Parameter(values['out_bias'].clone())
            self.shift_B = nn.Parameter(values['shift_B'].clone())
            self.shift_C = nn.Parameter(values['shift_C'].clone())
            self.shift_D = nn.Parameter(values['shift_D'].clone())
            self.log_dt = nn.Parameter(values['log_dt'].clone())
            self.inv_A_real = nn.Parameter(values['inv_A_real'].clone())
            self.A_imag = nn.Parameter(values['A_imag'].clone())
            self.diag_B = nn.Parameter(values['diag_B'].clone())
            self.diag_C = nn.Parameter(values['diag_C'].clone())
            self.diag_D = nn.Parameter(values['diag_D'].clone())

        def forward(self, u, shift0, diagonal0):
            # Precompute invariant coefficients
            # f001: constant 1.0
            c1 = torch.tensor(1.0, dtype=torch.float32, device=u.device)
            # f002: constant 2.0
            c2 = torch.tensor(2.0, dtype=torch.float32, device=u.device)

            # f003: exp(log_dt) -> [2]
            exp_log_dt = torch.exp(self.log_dt)
            # f004: unsqueeze(exp_log_dt, -1) -> [2, 1]
            exp_log_dt_u = exp_log_dt.unsqueeze(-1)

            # f005: exp(inv_A_real) -> [2, 2]
            exp_inv_A_real = torch.exp(self.inv_A_real)
            # f006: neg(exp_inv_A_real) -> [2, 2]
            neg_exp_inv_A_real = -exp_inv_A_real

            # f007: mul(neg_exp_inv_A_real, exp_log_dt_u) -> [2, 2]
            f007 = neg_exp_inv_A_real * exp_log_dt_u
            # f008: div(f007, c2) -> [2, 2]
            f008 = f007 / c2

            # f009: mul(A_imag, exp_log_dt_u) -> [2, 2]
            f009 = self.A_imag * exp_log_dt_u
            # f010: div(f009, c2) -> [2, 2]
            f010 = f009 / c2

            # f011: add(f008, c1) -> [2, 2]
            f011 = f008 + c1
            # f012: sub(c1, f008) -> [2, 2]
            f012 = c1 - f008
            # f013: neg(f010) -> [2, 2]
            f013 = -f010

            # f014: mul(f012, f012) -> [2, 2]
            f014 = f012 * f012
            # f015: mul(f013, f013) -> [2, 2]
            f015 = f013 * f013
            # f016: add(f014, f015) -> [2, 2]
            f016 = f014 + f015
            # f017: neg(f013) -> [2, 2]
            f017 = -f013

            # f018: mul(f011, f012) -> [2, 2]
            f018 = f011 * f012
            # f019: mul(f010, f017) -> [2, 2]
            f019 = f010 * f017
            # f020: sub(f018, f019) -> [2, 2]
            f020 = f018 - f019
            # f021: mul(f011, f017) -> [2, 2]
            f021 = f011 * f017
            # f022: mul(f010, f012) -> [2, 2]
            f022 = f010 * f012
            # f023: add(f021, f022) -> [2, 2]
            f023 = f021 + f022

            # f024: div(f020, f016) -> [2, 2] (ar)
            ar = f020 / f016
            # f025: div(f023, f016) -> [2, 2] (ai)
            ai = f023 / f016

            # f026: slice(diag_B, axis=2, start=0, stop=1) -> [2, 2, 1]
            f026 = self.diag_B[:, :, 0:1]
            # f027: squeeze(f026, -1) -> [2, 2]
            f027 = f026.squeeze(-1)
            # f028: mul(f027, exp_log_dt_u) -> [2, 2]
            f028 = f027 * exp_log_dt_u

            # f029: slice(diag_B, axis=2, start=1, stop=2) -> [2, 2, 1]
            f029 = self.diag_B[:, :, 1:2]
            # f030: squeeze(f029, -1) -> [2, 2]
            f030 = f029.squeeze(-1)
            # f031: mul(f030, exp_log_dt_u) -> [2, 2]
            f031 = f030 * exp_log_dt_u

            # f032: mul(f012, f012) -> [2, 2]
            f032 = f012 * f012
            # f033: mul(f013, f013) -> [2, 2]
            f033 = f013 * f013
            # f034: add(f032, f033) -> [2, 2]
            f034 = f032 + f033
            # f035: neg(f013) -> [2, 2]
            f035 = -f013

            # f036: mul(f028, f012) -> [2, 2]
            f036 = f028 * f012
            # f037: mul(f031, f035) -> [2, 2]
            f037 = f031 * f035
            # f038: sub(f036, f037) -> [2, 2]
            f038 = f036 - f037
            # f039: mul(f028, f035) -> [2, 2]
            f039 = f028 * f035
            # f040: mul(f031, f012) -> [2, 2]
            f040 = f031 * f012
            # f041: add(f039, f040) -> [2, 2]
            f041 = f039 + f040

            # f042: div(f038, f034) -> [2, 2] (br)
            br = f038 / f034
            # f043: div(f041, f034) -> [2, 2] (bi)
            bi = f041 / f034

            # f044: slice(diag_C, axis=2, start=0, stop=1) -> [2, 2, 1]
            f044 = self.diag_C[:, :, 0:1]
            # f045: squeeze(f044, -1) -> [2, 2] (cr)
            cr = f044.squeeze(-1)

            # f046: slice(diag_C, axis=2, start=1, stop=2) -> [2, 2, 1]
            f046 = self.diag_C[:, :, 1:2]
            # f047: squeeze(f046, -1) -> [2, 2] (ci)
            ci = f046.squeeze(-1)

            # f048: slice(diagonal0, axis=3, start=0, stop=1) -> [B, 2, 2, 1]
            f048 = diagonal0[:, :, :, 0:1]
            # f049: squeeze(f048, -1) -> [B, 2, 2] (r0)
            r0 = f048.squeeze(-1)

            # f050: slice(diagonal0, axis=3, start=1, stop=2) -> [B, 2, 2, 1]
            f050 = diagonal0[:, :, :, 1:2]
            # f051: squeeze(f050, -1) -> [B, 2, 2] (i0)
            i0 = f050.squeeze(-1)

            # Scan
            B, T, _ = u.shape
            y_list = []
            s = shift0
            r = r0
            i = i0

            for t in range(T):
                x = u[:, t, :] # [B, 2]

                # q_projection
                q = F.linear(x, self.q_weight, self.q_bias) # [B, 2]
                # k_projection
                k = F.linear(x, self.k_weight, self.k_bias) # [B, 2]
                # v_projection
                v = F.linear(x, self.v_weight, self.v_bias) # [B, 2]

                # shift_right: pad(s, [1, -1], constant, 0.0)
                # s is [B, 2, 4]. Pad last dim: left 1, right -1 (remove last)
                s_padded = F.pad(s, [1, -1], mode='constant', value=0.0) # [B, 2, 4]

                # f056: unsqueeze(k, -1) -> [B, 2, 1]
                k_u = k.unsqueeze(-1)
                # f057: mul(k_u, shift_B) -> [B, 2, 4]
                f057 = k_u * self.shift_B

                # shift_update: add(s_padded, f057) -> [B, 2, 4]
                s_new = s_padded + f057

                # f059: mul(s_new, shift_C) -> [B, 2, 4]
                f059 = s_new * self.shift_C
                # f060: sum(f059, dim=-1, keepdim=False) -> [B, 2]
                f060 = f059.sum(dim=-1)

                # f061: mul(k, shift_D) -> [B, 2]
                f061 = k * self.shift_D
                # f062: add(f060, f061) -> [B, 2]
                f062 = f060 + f061

                # value_gate: mul(f062, v) -> [B, 2]
                value_gate = f062 * v

                # f064: unsqueeze(value_gate, -1) -> [B, 2, 1]
                vg_u = value_gate.unsqueeze(-1)

                # f065: mul(r, ar) -> [B, 2, 2]
                f065 = r * ar
                # f066: mul(i, ai) -> [B, 2, 2]
                f066 = i * ai
                # f067: sub(f065, f066) -> [B, 2, 2]
                f067 = f065 - f066

                # f068: mul(r, ai) -> [B, 2, 2]
                f068 = r * ai
                # f069: mul(i, ar) -> [B, 2, 2]
                f069 = i * ar
                # f070: add(f068, f069) -> [B, 2, 2]
                f070 = f068 + f069

                # f071: mul(vg_u, br) -> [B, 2, 2]
                f071 = vg_u * br

                # diagonal_real_update: add(f067, f071) -> [B, 2, 2]
                r_new = f067 + f071

                # f073: mul(vg_u, bi) -> [B, 2, 2]
                f073 = vg_u * bi

                # diagonal_imag_update: add(f070, f073) -> [B, 2, 2]
                i_new = f070 + f073

                # f075: mul(r_new, cr) -> [B, 2, 2]
                f075 = r_new * cr
                # f076: mul(i_new, ci) -> [B, 2, 2]
                f076 = i_new * ci
                # f077: sub(f075, f076) -> [B, 2, 2]
                f077 = f075 - f076
                # f078: sum(f077, dim=-1, keepdim=False) -> [B, 2]
                f078 = f077.sum(dim=-1)

                # f079: constant 2.0
                c2_step = torch.tensor(2.0, dtype=torch.float32, device=u.device)
                # f080: mul(f078, c2_step) -> [B, 2]
                f080 = f078 * c2_step

                # f081: mul(value_gate, diag_D) -> [B, 2]
                f081 = value_gate * self.diag_D

                # f082: add(f080, f081) -> [B, 2]
                f082 = f080 + f081

                # query_gate: mul(q, f082) -> [B, 2]
                query_gate = q * f082

                # output_projection
                y_t = F.linear(query_gate, self.out_weight, self.out_bias) # [B, 2]

                y_list.append(y_t)

                # Update states
                s = s_new
                r = r_new
                i = i_new

            # Stack y
            y = torch.stack(y_list, dim=1) # [B, T, 2]

            # pair_final: stack(rf, if, dim=-1) -> [B, 2, 2, 2]
            diagonal_final = torch.stack([r, i], dim=-1)

            return y, s, diagonal_final

    return GatedStateBlock(values)
