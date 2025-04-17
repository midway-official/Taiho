#include "DNS.h"
#include "parallel.h"
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
        u_file << mesh.u_star;
        u_file.close();

        // 保存v场
        std::ofstream v_file(v_filename);
        if(!v_file) {
            throw std::runtime_error("无法创建文件: " + v_filename);
        }
        v_file << mesh.v_star;
        v_file.close();

        // 保存p场
        std::ofstream p_file(p_filename);
        if(!p_file) {
            throw std::runtime_error("无法创建文件: " + p_filename);
        }
        p_file << mesh.p_star;
        p_file.close();

        //std::cout << "进程 " << rank << " 的数据已保存到文件" << std::endl;
    }
    catch(const std::exception& e) {
        std::cerr << "保存数据时出错: " << e.what() << std::endl;
    }
}





int main(int argc, char* argv[]) 
{

    std::string mesh_folder;
    double mu = 1.0; 
    double dt;
    int timesteps;
    int n_x0,n_y0;
        if(argc == 5) {  // 增加一个参数
        // 命令行参数输入
        mesh_folder = argv[1];
        dt = std::stod(argv[2]);
        timesteps = std::stoi(argv[3]);
        mu = std::stod(argv[4]);  // 读取动力粘度
        std::cout << "从命令行读取参数:" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "时间步长: " << dt << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
        std::cout << "运动粘度: " << mu << std::endl;
    }
    else {
        // 手动输入
        std::cout << "网格文件夹路径:";
        std::cin >> mesh_folder;
        std::cout << "时间步长:";
        std::cin >> dt;
        std::cout << "时间步长数:";
        std::cin >> timesteps;
        std::cout << "运动粘度:";
        std::cin >> mu;
    }



    // 直接从文件夹加载网格
    Mesh mesh(mesh_folder);
    n_x0=mesh.nx;
    n_y0=mesh.ny;
    
    //建立u v p的方程
    Equation equ_u(mesh);
    Equation equ_v(mesh);
    Equation equ_p(mesh);
    
 
    auto start_time = chrono::steady_clock::now();  // 开始计时
    // 循环执行
    for (int i = 0; i <= timesteps; ++i) { 
       
       //记录上一个时间步长的u v
       mesh.u0 = mesh.u;
       mesh.v0 = mesh.v;
       //每步最大迭代次数
       int max_outer_iterations=100;
       //simple算法迭代
       for(int n=1;n<=max_outer_iterations;n++)
          {
          //1离散动量方程 
          momentum_function_unsteady(mesh,equ_u,equ_v,mu,dt);
          //2组合线性方程组
          equ_u.build_matrix();
          equ_v.build_matrix();
          
          
          //3求解线性方程组
          
          
        solve(equ_u,0.001,l2_norm_x,mesh.u);
        solve(equ_v,0.001,l2_norm_y,mesh.v);
          //4速度插值到面
          face_velocity(mesh ,equ_u);
          
          //5离散压力修正方程
          pressure_function(mesh,equ_p,equ_u);
          
          //6组合线性方程组
          equ_p.build_matrix();
          
          //7求解压力修正方程
          double epsilon_p=1e-5;
          
          solve(equ_p,0.001,l2_norm_p,mesh.p_prime);
          //8压力修正
          correct_pressure(mesh,equ_u);
          
          //9速度修正
          correct_velocity(mesh,equ_u);
          
          
          //10更新压力
          mesh.p=mesh.p_star;
          
          //收敛性判断
           std::cout << scientific 
          << " 轮数 " << n 
          << " u速度残差 " << setprecision(6) << (l2_norm_x)
          << " v速度残差 " << setprecision(6) << (l2_norm_y)
          << " 压力残差 " <<  setprecision(6) << (l2_norm_p)
          << "\n" <<  endl;
           if((l2_norm_x) < 1e-10 & (l2_norm_y) < 1e-10 & (l2_norm_p) < 1e-10)
           { 
           
            break;
           }
        
        }
    //时间推进
    mesh.u0 = mesh.u_star;
    mesh.v0 = mesh.v_star;
    
    
    //保存信息
    saveMeshData(mesh,0);
     // 返回到 result 文件夹
    
    }
 
    // 最后显示实际计算总耗时
    auto total_elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "\n计算完成 总耗时: " << total_elapsed_time << "秒" << std::endl;

}