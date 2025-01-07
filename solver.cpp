#include "DNS.h"
#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;

int main(int argc, char* argv[]) 
{

    std::string mesh_folder;
    double dt;
    int timesteps;
    int n_x0,n_y0;
    if(argc == 4) {
        // 命令行参数输入
        mesh_folder = argv[1];
        dt = std::stod(argv[2]);
        timesteps = std::stoi(argv[3]);
        std::cout << "从命令行读取参数:" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "时间步长: " << dt << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
    }
    else {
        // 手动输入
        std::cout << "网格文件夹路径:";
        std::cin >> mesh_folder;
        std::cout << "时间步长:";
        std::cin >> dt;
        std::cout << "时间步长数:";
        std::cin >> timesteps;
    }

    // 检查参数合法性
    if(dt <= 0 || timesteps <= 0) {
        std::cerr << "错误: 时间步长和步数必须为正数" << std::endl;
        return 1;
    }

    // 直接从文件夹加载网格
    Mesh mesh(mesh_folder);
    n_x0=mesh.nx;
    n_y0=mesh.ny;
    
    //建立u v p的方程
    Equation equ_u(mesh);
    Equation equ_v(mesh);
    Equation equ_p(mesh);
    
     // 创建顶层的 result 文件夹
    std::string result_folder = "result";
    if (!fs::exists(result_folder)) {
            fs::create_directory(result_folder);
            
        }

    // 切换到 result 文件夹
    fs::current_path(result_folder);
    auto start_time = chrono::steady_clock::now();  // 开始计时
    // 循环执行
    for (int i = 0; i <= timesteps; ++i) { 
        string folder_name = to_string(i);
        if (!fs::exists(folder_name)) {
            fs::create_directory(folder_name);
            cout << "结果保存于: " << folder_name << endl;
        }
        cout<<"时间步长 "<< i <<std::endl;
        // 切换到当前编号文件夹
       fs::current_path(folder_name);
       //记录上一个时间步长的u v
       mesh.u0 = mesh.u;
       mesh.v0 = mesh.v;
       //每步最大迭代次数
       int max_outer_iterations=300;
       //simple算法迭代
       for(int n=1;n<=max_outer_iterations;n++)
          {
          //1离散动量方程 
          movement_function(mesh,equ_u,equ_v,100000);
          //2组合线性方程组
          equ_u.build_matrix();
          equ_v.build_matrix();
          
          
          //3求解线性方程组
          double epsilon_uv=0.1;
          
          solve(equ_u, epsilon_uv, l2_norm_x, mesh.u);
          solve(equ_v, epsilon_uv, l2_norm_y, mesh.v);
          
          //4速度插值到面
          face_velocity(mesh ,equ_u);
          
          //5离散压力修正方程
          pressure_function(mesh,equ_p,equ_u);
         
          //6组合线性方程组
          equ_p.build_matrix();
          //7求解压力修正方程
          double epsilon_p=1e-5;
          
          solve(equ_p, epsilon_p, l2_norm_p, mesh.p_prime);
          
          
          //8压力修正
          correct_pressure(mesh,equ_u);
          
          //9速度修正
          correct_velocity(mesh,equ_u);
          
          //10更新压力
          mesh.p=mesh.p_star;
          
          //收敛性判断
           std::cout << scientific 
          << " 轮数 " << n 
          << " u速度残差 " << setprecision(6) << (l2_norm_x/(n_x0*n_y0))
          << " v速度残差 " << setprecision(6) << (l2_norm_y/(n_x0*n_y0))
          << " 压力残差 " <<  setprecision(6) << (l2_norm_p/(n_x0*n_y0))
          << "\n" <<  endl;
           if((l2_norm_x/(n_x0*n_y0)) < 1e-8 & (l2_norm_y/(n_x0*n_y0)) < 1e-8 & (l2_norm_p/(n_x0*n_y0)) < 1e-9)
           { 
           
            break;
           }
        
        }
    //显式时间推进
    mesh.u = mesh.u0 + dt*mesh.u;
    mesh.v = mesh.v0 + dt*mesh.v;
    
    // 显示进度条
    auto elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    show_progress_bar(i, timesteps, elapsed_time);
    //保存信息
    post_processing(mesh,n_x0,n_y0,a);
     // 返回到 result 文件夹
    fs::current_path("..");
    }
        // 返回到程序主目录
    fs::current_path("..");
    // 最后显示实际计算总耗时
    auto total_elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "\n计算完成 总耗时: " << total_elapsed_time << "秒" << std::endl;

}