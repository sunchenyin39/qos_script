import numpy as np
import matplotlib.pyplot as plt
import fit

data = np.load('temp.npy', allow_pickle=True)
data = data.tolist()
# print(data['slow']['primitive'])

# 散点
gate_num_scatter_reference = []
gate_num_scatter = []
fidelity_scatter_reference = []
fidelity_scatter = []
for i in range(len(data['slow']['primitive'])):
    if data['slow']['primitive'][i][3] == 0:
        gate_num_scatter_reference.append(data['slow']['primitive'][i][1])
        fidelity_scatter_reference.append(data['slow']['primitive'][i][4])
    else:
        gate_num_scatter.append(data['slow']['primitive'][i][1])
        fidelity_scatter.append(data['slow']['primitive'][i][4])

# 平均点(字典中已平均的)
gate_num_line_reference = []
gate_num_line = []
fidelity_line_reference = []
fidelity_line = []
for i in range(len(data['slow']['rb_data'])):
    gate_num_line_reference.append(data['slow']['rb_data'][i][1])
    gate_num_line.append(data['slow']['rb_data'][i][1])
    fidelity_line_reference.append(data['slow']['rb_data'][i][2])
    fidelity_line.append(data['slow']['rb_data'][i][3])

# 平均点(自己平均的)
fidelity_line_reference_average = np.zeros(len(gate_num_line))
fidelity_line_average = np.zeros(len(gate_num_line))
for i in range(len(data['slow']['primitive'])):
    if data['slow']['primitive'][i][3] == 0:
        fidelity_line_reference_average[data['slow']['primitive'][i][1]]+=data['slow']['primitive'][i][4] 
    else:
        fidelity_line_average[data['slow']['primitive'][i][1]]+=data['slow']['primitive'][i][4] 
fidelity_line_reference_average=fidelity_line_reference_average/30.0
fidelity_line_average=fidelity_line_average/30.0

# 拟合
parameters_ref, err_ref = fit.exp_Fit_1D(gate_num_line_reference, fidelity_line_reference_average)
Pref=np.exp(parameters_ref[1])
gate_num_line_reference_fit=np.arange(0, 13, 0.2)
fidelity_line_reference_average_fit = parameters_ref[0]*np.exp(parameters_ref[1]*gate_num_line_reference_fit)+parameters_ref[2]

parameters, err = fit.exp_Fit_1D(gate_num_line, fidelity_line_average)
Pgate=np.exp(parameters_ref[1])
gate_num_line_fit=np.arange(0, 13, 0.2)
fidelity_line_average_fit = parameters[0]*np.exp(parameters[1]*gate_num_line_fit)+parameters[2]

# 画图
type1=plt.scatter(gate_num_scatter_reference,
            fidelity_scatter_reference, s=0.5, c='blue')
type2=plt.scatter(gate_num_scatter, fidelity_scatter, s=0.5, c='red')
# plt.plot(gate_num_line_reference, fidelity_line_reference, c='blue')
# plt.plot(gate_num_line, fidelity_line, c='red')
# plt.plot(gate_num_line_reference, fidelity_line_reference_average, c='yellow')
# plt.plot(gate_num_line, fidelity_line_average, c='green')
plt.plot(gate_num_line_reference_fit, fidelity_line_reference_average_fit, c='blue')
plt.plot(gate_num_line_fit, fidelity_line_average_fit, c='red')
plt.title("RB")
plt.ylabel("sequence fidelity")
plt.xlabel("numbers of gate")
plt.legend((type1,type2),("reference","interleaved"))
plt.show()
