clear; close all;
% 参数设置
sigma = [5,5,5];      %[X轴σ, Y轴σ, Z轴σ]
k_size = [21,21,21];  %核尺寸需为奇数
output_file = 'gausskernel555.mat';

% 生成并保存核
create3DGauss(sigma, k_size, output_file);

% 验证核属性
load(output_file);
disp(['核尺寸：' num2str(size(gauss3d))]);
disp(['元素和：' num2str(sum(gauss3d(:))) '（应≈1）']);

function create3DGauss(sigma_xyz, kernel_size, save_path)
% 创建可自定义参数的三维高斯核
% 输入参数：
%   sigma_xyz - 三维标准差向量 [σ_x, σ_y, σ_z]
%   kernel_size - 核尺寸向量 [size_x, size_y, size_z]
%   save_path - 保存路径（如'my_gauss3d.mat'）

% 生成三维坐标网格
[x,y,z] = ndgrid(...
    linspace(-(kernel_size(1)-1)/2, (kernel_size(1)-1)/2, kernel_size(1)),...
    linspace(-(kernel_size(2)-1)/2, (kernel_size(2)-1)/2, kernel_size(2)),...
    linspace(-(kernel_size(3)-1)/2, (kernel_size(3)-1)/2, kernel_size(3)));

% 计算三维高斯分布
exponent = -(x.^2/(2*sigma_xyz(1)^2) + y.^2/(2*sigma_xyz(2)^2) + z.^2/(2*sigma_xyz(3)^2));
gauss3d = exp(exponent);

% 归一化处理
gauss3d = gauss3d / sum(gauss3d(:));

% 保存数据
save(save_path, 'gauss3d', '-v7.3');
disp(['三维高斯核已保存至：' save_path]);
end