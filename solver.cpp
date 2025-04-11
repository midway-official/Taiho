#include "DNS.h"
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;
void saveMeshData(const Mesh& mesh, int rank, const std::string& timestep_folder = "") {
    // 创建文件名
    std::string u_filename = "u_" + std::to_string(rank) + ".dat";
    std::string v_filename = "v_" + std::to_string(rank) + ".dat";
    std::string p_filename = "p_" + std::to_string(rank) + ".dat";
    
    // 如果提供了时间步文件夹，添加到路径中
    if(!timestep_folder.empty()) {
        if (!fs::exists(timestep_folder)) {
            fs::create_directory(timestep_folder);
        }
        u_filename = timestep_folder + "/" + u_filename;
        v_filename = timestep_folder + "/" + v_filename;
        p_filename = timestep_folder + "/" + p_filename;
    }

    try {
        // 保存u场
        std::ofstream u_file(u_filename);
        if(!u_file) {
            throw std::runtime_error("无法创建文件: " + u_filename);
        }
        u_file << mesh.u;
        u_file.close();

        // 保存v场
        std::ofstream v_file(v_filename);
        if(!v_file) {
            throw std::runtime_error("无法创建文件: " + v_filename);
        }
        v_file << mesh.v;
        v_file.close();

        // 保存p场
        std::ofstream p_file(p_filename);
        if(!p_file) {
            throw std::runtime_error("无法创建文件: " + p_filename);
        }
        p_file << mesh.p;
        p_file.close();

        //std::cout << "进程 " << rank << " 的数据已保存到文件" << std::endl;
    }
    catch(const std::exception& e) {
        std::cerr << "保存数据时出错: " << e.what() << std::endl;
    }
}


int main(int argc, char* argv[]) 
{
    // 获取输入参数
    std::string mesh_folder;
    double dt;
    int timesteps;
   
    double mu;
    
    if(argc == 5) {
        // 命令行参数输入
        mesh_folder = argv[1];
        dt = std::stod(argv[2]);
        timesteps = std::stoi(argv[3]);
        mu= std::stod(argv[4]);
        
        std::cout << "从命令行读取参数:" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "时间步长: " << dt << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
        
        std::cout << "粘度: " << mu << std::endl;

    }
    else {
        // 手动输入
        std::cout << "网格文件夹路径:";
        std::cin >> mesh_folder;
        std::cout << "时间步长:";
        std::cin >> dt;
        std::cout << "时间步长数:";
        std::cin >> timesteps;
        
        std::cout << "粘度:";
        std::cin >> mu;
    }



    // 加载原始网格
    Mesh mesh(mesh_folder);
    
    // ... 后续的计算过程 ...
   int nx0,ny0;
   nx0=mesh.nx;
   ny0=mesh.ny;
    //建立u v p的方程
    Equation equ_u(mesh);
    Equation equ_v(mesh);
    Equation equ_p(mesh);
   double l2x = 0.0, l2y = 0.0, l2p = 0.0;
   auto start_time0 = chrono::steady_clock::now();  // 开始计时
    for (int i = 0; i <= timesteps; ++i) { 
        
        cout<<"时间步长 "<< i <<std::endl;
        // 切换到当前编号文件夹
      
       //记录上一个时间步长的u v
      
       int max_outer_iterations=25;
           //simple算法迭代
  
        double init_l2_norm_x = -1.0;
       double init_l2_norm_y = -1.0;
       double init_l2_norm_p = -1.0;
    for(int n=1;n<=max_outer_iterations;n++) {
       
        //1离散动量方程 
        double l2_norm_x, l2_norm_y;
        
       
        movement_function_unsteady(mesh,equ_u,equ_v,mu,dt);
        equ_u.build_matrix();
        equ_v.build_matrix();


    
        //3求解线性方程组
        double epsilon_uv=0.0000001;
        solve(equ_u,epsilon_uv,l2_norm_x,mesh.u);
        solve(equ_v,epsilon_uv,l2_norm_y,mesh.v);
        
        
        

      
        
       
        //4速度插值到面
        face_velocity(mesh ,equ_u);
        
        
        
        double epsilon_p=1e-8;
       
        pressure_function(mesh, equ_p, equ_u);
       
        // 重新更新源项并重建矩阵
        equ_p.build_matrix();
        //求解
        solve(equ_p,epsilon_p,l2_norm_p,mesh.p_prime);
        //8压力修正
        correct_pressure(mesh,equ_u);
       
        //9速度修正
        correct_velocity(mesh,equ_u);
       
        
        
        //10更新压力
        mesh.p=mesh.p_star;
       
        
        
     
        
       
        
       
       // 记录初始残差（仅第一次迭代）
if (n == 1) {
    init_l2_norm_x = l2_norm_x;
    init_l2_norm_y = l2_norm_y;
    init_l2_norm_p = l2_norm_p;
}

// 避免除以 0（数值健壮性）
double norm_res_x = (init_l2_norm_x > 1e-200) ? l2_norm_x / init_l2_norm_x : 0.0;
double norm_res_y = (init_l2_norm_y > 1e-200) ? l2_norm_y / init_l2_norm_y : 0.0;
double norm_res_p = (init_l2_norm_p > 1e-200) ? l2_norm_p / init_l2_norm_p : 0.0;
   
// 只在主进程(rank=0)打印残差信息

    std::cout << scientific 
              << "时间步: " << i 
              << " 迭代轮数: " << n 
              <<"  归一化残差："
              << " u: " << setprecision(4) << norm_res_x
              << " v: " << setprecision(4) << norm_res_y
              << " p " << setprecision(4) << norm_res_p
              <<"  全局残差："
              << " u: " << setprecision(4) << l2_norm_x
              << " v: " << setprecision(4) << l2_norm_y
              << " p " << setprecision(4) << l2_norm_p
              << std::endl;


// 检查收敛性
int local_converged = (norm_res_x < 1e-1) && 
                      (norm_res_y < 1e-1) && 
                      (norm_res_p < 1e-3);


if(local_converged) {
    
        std::cout << "所有进程达到收敛条件" << std::endl;
    
    break;
}
    
    }
    //时间推进
     post_processing(mesh);
    mesh.u0 = mesh.u_star;
    mesh.v0 = mesh.v_star;
    }
    
   
    // 最后显示实际计算总耗时
    auto total_elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time0).count();
    std::cout << "\n计算完成 总耗时: " << total_elapsed_time << "秒" << std::endl;
  
    
   
     
    return 0;
}