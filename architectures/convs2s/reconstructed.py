import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, D=3, C=4, K=3):
        super().__init__()
        self.D = D
        self.C = C
        self.K = K
        
        self.p = nn.ParameterDict({
            'enc_in_w': nn.Parameter(torch.zeros(C, D, dtype=torch.float64)),
            'enc_in_b': nn.Parameter(torch.zeros(C, dtype=torch.float64)),
            'enc_conv_w': nn.Parameter(torch.zeros(2*C, C, K, dtype=torch.float64)),
            'enc_conv_b': nn.Parameter(torch.zeros(2*C, dtype=torch.float64)),
            'enc_out_w': nn.Parameter(torch.zeros(D, C, dtype=torch.float64)),
            'enc_out_b': nn.Parameter(torch.zeros(D, dtype=torch.float64)),
            'dec_in_w': nn.Parameter(torch.zeros(C, D, dtype=torch.float64)),
            'dec_in_b': nn.Parameter(torch.zeros(C, dtype=torch.float64)),
            'dec_conv_w': nn.Parameter(torch.zeros(2*C, C, K, dtype=torch.float64)),
            'dec_conv_b': nn.Parameter(torch.zeros(2*C, dtype=torch.float64)),
            'attn_in_w': nn.Parameter(torch.zeros(D, C, dtype=torch.float64)),
            'attn_in_b': nn.Parameter(torch.zeros(D, dtype=torch.float64)),
            'attn_out_w': nn.Parameter(torch.zeros(C, D, dtype=torch.float64)),
            'attn_out_b': nn.Parameter(torch.zeros(C, dtype=torch.float64))
        })

    def _linear(self, x, w, b):
        # x: [..., in_features], w: [out_features, in_features], b: [out_features]
        # result: [..., out_features]
        return torch.matmul(x, w.t()) + b

    def _conv(self, x, w, b, left):
        # x: [B, L, C], w: [2*C, C, K], b: [2*C], left: int
        # result: [B, L, 2*C]
        B, L, C = x.shape
        K = self.K
        
        # Pad x with zeros on the left and right
        # Total padding = K - 1
        # Left padding = left
        # Right padding = (K - 1) - left
        left_pad = left
        right_pad = (K - 1) - left
        
        # x_padded: [B, L + K - 1, C]
        x_padded = torch.nn.functional.pad(x, (0, 0, left_pad, right_pad))
        
        # Unfold to get patches
        # We want output at time t to use x[t + r - left] for r in 0..K-1
        # This corresponds to a window of size K starting at index t in the original sequence
        # In the padded sequence, the window starting at index t in original corresponds to index t + left_pad in padded
        # So we extract windows of size K starting at indices left_pad, left_pad+1, ..., left_pad+L-1
        
        # Using unfold: x_padded.unfold(1, K, 1) gives [B, L + K - 1 - K + 1, C, K] = [B, L, C, K]
        # Wait, unfold(1, K, 1) on a tensor of size L+K-1 gives L windows.
        # The i-th window (0-indexed) starts at i. We want the window starting at left_pad + t for t in 0..L-1.
        # So we take windows from index left_pad to left_pad + L - 1.
        
        # Let's just do it manually or with a loop for clarity and correctness, or use unfold properly.
        # x_padded shape: [B, L + K - 1, C]
        # We want to extract L windows of size K.
        # Window t (0-indexed) should correspond to original time t.
        # Original time t corresponds to padded index t + left_pad.
        # So window t is x_padded[:, t + left_pad : t + left_pad + K, :]
        
        # Using unfold:
        # x_padded.unfold(1, K, 1) has shape [B, L, C, K] if L + K - 1 - K + 1 = L.
        # The j-th window in unfold starts at index j.
        # We want the window starting at index left_pad + t.
        # So we need to select windows from index left_pad to left_pad + L - 1.
        # But unfold gives windows starting at 0, 1, ..., L + K - 2.
        # We want windows starting at left_pad, left_pad+1, ..., left_pad+L-1.
        # This is a slice of the unfold result.
        
        # Actually, let's just use a loop or manual indexing to be safe.
        # Or better, use F.conv1d with appropriate padding.
        # F.conv1d expects [B, C_in, L]. Our x is [B, L, C].
        # Let's transpose x to [B, C, L].
        x_t = x.transpose(1, 2) # [B, C, L]
        
        # F.conv1d with padding
        # We want cross-correlation: sum_r W[o, i, r] * x[t + r - left]
        # This is equivalent to convolution with a flipped kernel if we use standard conv.
        # But the YAML defines it as cross_correlation.
        # Let's stick to the manual definition to avoid confusion.
        
        # Let's implement the cross-correlation directly.
        # out[b, t, o] = b[o] + sum_i sum_r W[o, i, r] * x[b, t + r - left, i]
        
        # We can do this by padding x and then using unfold or manual indexing.
        # Let's use manual indexing for clarity.
        
        # Create output tensor
        out = torch.zeros(B, L, 2*C, dtype=x.dtype, device=x.device)
        
        # Add bias
        out += b[None, None, :]
        
        # Compute convolution
        # For each output channel o, input channel i, and kernel position r
        # We can vectorize this.
        
        # Let's pad x first.
        # x_padded[b, t, i] = x[b, t - left_pad, i] if 0 <= t - left_pad < L else 0
        # This is equivalent to shifting x by left_pad positions to the right.
        
        # Let's create a padded version of x.
        # x_padded shape: [B, L + K - 1, C]
        # x_padded[:, t, :] = x[:, t - left_pad, :] if 0 <= t - left_pad < L else 0
        
        # We can do this by creating a zero tensor and copying x into it.
        x_padded = torch.zeros(B, L + K - 1, C, dtype=x.dtype, device=x.device)
        # Copy x into x_padded starting at index left_pad
        x_padded[:, left_pad : left_pad + L, :] = x
        
        # Now, for each t in 0..L-1, we want to use x_padded[:, t : t + K, :]
        # This is a window of size K.
        # We can use unfold to get all windows.
        # x_padded.unfold(1, K, 1) gives [B, L, C, K]
        # The t-th window is x_padded[:, t : t + K, :]
        
        windows = x_padded.unfold(1, K, 1) # [B, L, C, K]
        
        # Now we need to compute: sum_i sum_r W[o, i, r] * windows[b, t, i, r]
        # This is a tensor contraction.
        # W shape: [2*C, C, K]
        # windows shape: [B, L, C, K]
        
        # We can reshape windows to [B, L, C*K] and W to [2*C, C*K]
        # Then do a matrix multiplication.
        
        windows_flat = windows.reshape(B, L, C * K) # [B, L, C*K]
        W_flat = w.reshape(2*C, C * K) # [2*C, C*K]
        
        # out[b, t, o] = sum_{i,r} W_flat[o, i*K+r] * windows_flat[b, t, i*K+r]
        # This is a batched matrix multiplication.
        # windows_flat: [B, L, C*K]
        # W_flat: [2*C, C*K]
        # We want [B, L, 2*C]
        
        # We can do: torch.matmul(windows_flat, W_flat.t())
        # windows_flat: [B, L, C*K]
        # W_flat.t(): [C*K, 2*C]
        # Result: [B, L, 2*C]
        
        conv_out = torch.matmul(windows_flat, W_flat.t()) # [B, L, 2*C]
        
        out += conv_out
        
        return out

    def _glu(self, a):
        # a: [..., 2*C]
        # result: [..., C]
        C = self.C
        a1 = a[..., :C]
        a2 = a[..., C:]
        return a1 * torch.sigmoid(a2)

    def forward(self, source_embedding, target_embedding, source_padding_mask):
        # source_embedding: [B, S, D]
        # target_embedding: [B, T, D]
        # source_padding_mask: [B, S] bool
        
        B, S, D = source_embedding.shape
        T = target_embedding.shape[1]
        
        # 1. Encoder Block
        # e0 = linear(source_embedding, enc_in_w, enc_in_b)
        e0 = self._linear(source_embedding, self.p['enc_in_w'], self.p['enc_in_b']) # [B, S, C]
        
        # em = where(source_padding_mask[:,:,None], 0, e0)
        mask_3d = source_padding_mask.unsqueeze(-1) # [B, S, 1]
        em = torch.where(mask_3d, torch.zeros_like(e0), e0) # [B, S, C]
        
        # eg = glu(conv(em, enc_conv_w, enc_conv_b, left=floor((K-1)/2)))
        left_pad = (self.K - 1) // 2
        conv_out_enc = self._conv(em, self.p['enc_conv_w'], self.p['enc_conv_b'], left_pad) # [B, S, 2*C]
        eg = self._glu(conv_out_enc) # [B, S, C]
        
        # eh = (eg + e0) / sqrt(2)
        eh = (eg + e0) / math.sqrt(2) # [B, S, C]
        
        # 2. Encoder Memory
        # encoder = where(source_padding_mask[:,:,None], 0, linear(eh, enc_out_w, enc_out_b))
        enc_proj = self._linear(eh, self.p['enc_out_w'], self.p['enc_out_b']) # [B, S, D]
        encoder = torch.where(mask_3d, torch.zeros_like(enc_proj), enc_proj) # [B, S, D]
        
        # values = (encoder + source_embedding) / sqrt(2)
        values = (encoder + source_embedding) / math.sqrt(2) # [B, S, D]
        
        # 3. Decoder Block
        # d0 = linear(target_embedding, dec_in_w, dec_in_b)
        d0 = self._linear(target_embedding, self.p['dec_in_w'], self.p['dec_in_b']) # [B, T, C]
        
        # decoder_glu = glu(conv(d0, dec_conv_w, dec_conv_b, left=K-1))
        left_pad_dec = self.K - 1
        conv_out_dec = self._conv(d0, self.p['dec_conv_w'], self.p['dec_conv_b'], left_pad_dec) # [B, T, 2*C]
        decoder_glu = self._glu(conv_out_dec) # [B, T, C]
        
        # 4. Attention
        # q = (linear(decoder_glu, attn_in_w, attn_in_b) + target_embedding) / sqrt(2)
        q_proj = self._linear(decoder_glu, self.p['attn_in_w'], self.p['attn_in_b']) # [B, T, D]
        q = (q_proj + target_embedding) / math.sqrt(2) # [B, T, D]
        
        # logits[b,t,s] = sum_d q[b,t,d] * encoder[b,s,d]
        # q: [B, T, D], encoder: [B, S, D]
        # logits: [B, T, S]
        logits = torch.bmm(q, encoder.transpose(1, 2)) # [B, T, S]
        
        # replace masked logits with -inf
        # source_padding_mask: [B, S] bool, True excludes
        # We need to mask the S dimension of logits.
        # logits: [B, T, S]
        # mask: [B, 1, S] to broadcast over T
        mask_2d = source_padding_mask.unsqueeze(1) # [B, 1, S]
        logits = torch.where(mask_2d, torch.full_like(logits, float('-inf')), logits)
        
        # attention = softmax(logits, axis=source S), subtract row maximum before exp
        # Standard softmax with -inf handling
        attention = F.softmax(logits, dim=-1) # [B, T, S]
        
        # 5. Context
        # n[b] = sum_s (not source_padding_mask[b,s])
        # source_padding_mask is True for padding, so we want count of False.
        valid_mask = ~source_padding_mask # [B, S] bool
        n = valid_mask.sum(dim=-1, keepdim=True).to(dtype=source_embedding.dtype) # [B, 1]
        
        # context[b,t,d] = sqrt(n[b]) * sum_s attention[b,t,s] * values[b,s,d]
        # attention: [B, T, S], values: [B, S, D]
        # context: [B, T, D]
        context_unscaled = torch.bmm(attention, values) # [B, T, D]
        context = context_unscaled * torch.sqrt(n).unsqueeze(-1) # [B, T, D]
        
        # 6. Residual
        # attended = (linear(context, attn_out_w, attn_out_b) + decoder_glu) / sqrt(2)
        ctx_proj = self._linear(context, self.p['attn_out_w'], self.p['attn_out_b']) # [B, T, C]
        attended = (ctx_proj + decoder_glu) / math.sqrt(2) # [B, T, C]
        
        # output = (attended + d0) / sqrt(2)
        output = (attended + d0) / math.sqrt(2) # [B, T, C]
        
        return {
            'encoder': encoder,
            'values': values,
            'decoder_glu': decoder_glu,
            'attention': attention,
            'context': context,
            'output': output
        }