import torch
import torchdiffeq
import torch.nn as nn
import lib.gamma as gamma

def belta(result_all,fractionalorder,series_order):
    fractionalorder_plus = fractionalorder+1
    fractionalorder_plus = fractionalorder_plus.clone().detach().to('cuda')
    belt_result = gamma.gamma_all(fractionalorder_plus,fractionalorder_plus)/(gamma.gamma_all(series_order+1,series_order+1)*gamma.gamma_all(fractionalorder-series_order+1,fractionalorder-series_order+1))
    result_all = result_all+belt_result
    return result_all


class FCM_series_1(torch.nn.Module):
    def __init__(self,  args,num,A_y_list, ww,    alpha,fract,train_init,lambd,l):
        super(FCM_series_1, self).__init__()

        self.lambd = lambd
        self.l = l
        self.args = args
        self.WW = ww
        self.alpha = alpha
        self.B_y_list = torch.zeros(4, args.batch_size, args.lag-1,args.num_nodes, 1).to('cuda')
        self.fract=fract
        self.num = num
        self.A_y_list = A_y_list
        self.num_b = 0
        self.belta_result = 0
        self.train_init = train_init
        self.A_N_OLD = self.train_init[..., :self.args.lag-1, 1].permute(2, 0, 1).unsqueeze(3)
        self.A_0_NEW = self.train_init[..., 1:self.args.lag, 1].permute(2, 0, 1).unsqueeze(3)

        A = self.A_N_OLD
        v_tmp_0 = A.repeat_interleave(self.args.num_nodes, dim=2)
        v_0 = v_tmp_0.view(self.args.lag-1,self.args.batch_size, self.args.num_nodes, self.args.num_nodes,self.args.init_len).contiguous()
        v_final_tmp_1 = torch.mul(self.WW, v_0)
        output_final_A = torch.sum(v_final_tmp_1[:, :, :, :], dim=2)
        Tanh = nn.Tanh()
        output_final_A = Tanh((-1)* output_final_A)
        self.A_N_OLD = output_final_A



        A = self.A_0_NEW
        v_tmp_0 = A.repeat_interleave(self.args.num_nodes, dim=2)  # 初始时刻值
        v_0 = v_tmp_0.view(self.args.lag-1,self.args.batch_size, self.args.num_nodes, self.args.num_nodes,
                           self.args.init_len).contiguous()  # 原来是16行30列，v_tmp_0是将16行30列中的每一行复制16次，
        v_final_tmp_1 = torch.mul(self.WW, v_0)
        output_final_A = torch.sum(v_final_tmp_1[:, :, :, :], dim=2)
        Tanh = nn.Tanh()
        output_final_A = Tanh((-1)* output_final_A)
        self.A_0_NEW = output_final_A




    def D(a, num,output_final_list,args):
        num=0
        def binomial_coeffs(p, k):
            y = (-1) ** k * gamma.gamma_all(p + 1, p + 1) / (
                        gamma.gamma_all(k + 1, k + 1) * gamma.gamma_all(p - k + 1, p - k + 1))
            return y
        h = 1 / 3
        sum_GL = 0
        for m in range(num+1):
            m_tmp = torch.full((args.lag-1,1),m).to('cuda')
            binomial_coeffs_tmp = binomial_coeffs(a, m_tmp)
            binomial_coeffs_tmp = binomial_coeffs_tmp.unsqueeze(0).unsqueeze(2)
            binomial_coeffs_tmp = binomial_coeffs_tmp.repeat(args.batch_size,1,args.num_nodes,1)
            sum_GL += output_final_list[num - m ,...] * binomial_coeffs_tmp
        a = a.unsqueeze(0).unsqueeze(2)
        a = a.repeat(args.batch_size,1,args.num_nodes,1)
        return sum_GL/(h**a)


    def __call__(self, t, A):

        A1=A[0]
        A = A1
        v_tmp_0 = A.repeat_interleave(self.args.num_nodes, dim=2)
        v_0 = v_tmp_0.view(self.args.lag-1,self.args.batch_size, self.args.num_nodes, self.args.num_nodes,
                           self.args.init_len).contiguous()
        v_final_tmp_1 = torch.mul(self.WW, v_0)

        output_final_A = torch.sum(v_final_tmp_1[:, :, :, :], dim=2)
        Tanh = nn.Tanh()
        output_final_A = Tanh((-1)* output_final_A)


        self.A_y_list[self.num,...] = output_final_A.transpose(0,1)
        output_final_A = FCM_series_1.D(self.fract, self.num, self.A_y_list,self.args)
        belta_result = 0
        belta_result = torch.tensor(belta_result).unsqueeze(0).unsqueeze(1).expand(self.args.lag-1, 1).to('cuda')

        for num_gamma in range(0, 4, 1):
            num_gamma = torch.tensor(num_gamma).unsqueeze(0).unsqueeze(1).expand(self.args.lag-1, 1).to('cuda')
            belta_result = belta(belta_result, self.alpha, num_gamma)
        c_alpha = self.alpha
        all_ones = torch.ones(self.args.lag-1, 1).to('cuda')
        tmp1 = gamma.gamma_all(c_alpha + 1,c_alpha + 1).unsqueeze(1).unsqueeze(2).expand(self.args.lag-1,self.args.batch_size,self.args.num_nodes,1)
        tmp2 = gamma.gamma_all(all_ones,all_ones).unsqueeze(1).unsqueeze(2).expand(self.args.lag-1,self.args.batch_size,self.args.num_nodes,1)
        tmp3 = gamma.gamma_all(c_alpha + 1,c_alpha + 1).unsqueeze(1).unsqueeze(2).expand(self.args.lag-1,self.args.batch_size,self.args.num_nodes,1)
        tmp4 = gamma.gamma_all((3 + 1)*all_ones,(3 + 1)*all_ones).unsqueeze(1).unsqueeze(2).expand(self.args.lag-1,self.args.batch_size,self.args.num_nodes,1)
        tmp5 = gamma.gamma_all(c_alpha - 3 + 1,c_alpha - 3 + 1).unsqueeze(1).unsqueeze(2).expand(self.args.lag-1,self.args.batch_size,self.args.num_nodes,1)
        constant = self.l.repeat(1, self.args.batch_size, 6, 1)*(((1) * tmp1 / (tmp2 * tmp3)) * self.A_0_NEW +
                    ((1) * tmp3 / (tmp4 * tmp5 ))* self.A_N_OLD) / (1 / 3)
        constant = constant.transpose(1, 0)
        if self.num ==11:
            self.num ==11
        else:
            self.num += 1
        if self.num_b==3:
            self.num_b=3
        else:
            self.num_b+=1
        output_final_all = output_final_A
        belta_result_tmp = belta_result.unsqueeze(0).unsqueeze(2).expand(self.args.batch_size,self.args.lag-1,self.args.num_nodes,1)
        output_final_all = (self.lambd.repeat(1, self.args.batch_size, 6, 1).transpose(1,0)/belta_result_tmp)*(2*output_final_all+constant)
        return tuple(output_final_all)





def cdeint_gde_dev( z0, args, num,A_y_list, t,method,atol,rtol,ww,     alpha ,fract, train_init,lambd,l,adjoint=True, **kwargs):

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = FCM_series_1( args,num,A_y_list,ww=ww,       alpha =alpha,fract=fract,train_init = train_init,lambd = lambd,l = l)

    init0 = (z0,)
    out = odeint(func=vector_field, y0=init0, t=t, method=method,atol=atol,rtol=rtol)
    out1 = out[0]
    return out1