#include "DNS.h"
#include <filesystem>
#include <chrono>
#include "parallel.h"
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Dense>
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

        std::cout << "进程 " << rank << " 的数据已保存到文件" << std::endl;
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
    int n_splits;  // 并行计算线程数

    if(argc == 5) {
        // 命令行参数输入
        mesh_folder = argv[1];
        dt = std::stod(argv[2]);
        timesteps = std::stoi(argv[3]);
        n_splits = std::stoi(argv[4]);
        std::cout << "从命令行读取参数:" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "时间步长: " << dt << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
        std::cout << "并行线程数: " << n_splits << std::endl;
    }
    else {
        // 手动输入
        std::cout << "网格文件夹路径:";
        std::cin >> mesh_folder;
        std::cout << "时间步长:";
        std::cin >> dt;
        std::cout << "时间步长数:";
        std::cin >> timesteps;
        std::cout << "并行线程数:";
        std::cin >> n_splits;
    }

    // 检查参数合法性
    if(dt <= 0 || timesteps <= 0 || n_splits <= 0) {
        std::cerr << "错误: 时间步长、步数和并行线程数必须为正数" << std::endl;
        return 1;
    }

    // 加载原始网格
    Mesh original_mesh(mesh_folder);
    
    // 垂直分割网格
    std::vector<Mesh> sub_meshes = splitMeshVertically(original_mesh, n_splits);
    
    // 打印分割信息
    std::cout << "网格已分割为 " << n_splits << " 个子网格:" << std::endl;
    for(int i = 0; i < sub_meshes.size(); i++) {
        std::cout << "子网格 " << i << " 尺寸: " 
                  << sub_meshes[i].nx << "x" << sub_meshes[i].ny << std::endl;
    }
    
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
     MPI_Barrier(MPI_COMM_WORLD);
    // 获取进程信息
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 检查进程数是否匹配
    if(num_procs != n_splits) {
        if(rank == 0) {
            std::cerr << "错误: MPI进程数 (" << num_procs 
                      << ") 与指定的并行线程数 (" << n_splits 
                      << ") 不匹配" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // 每个进程获取对应的子网格
    Mesh mesh = sub_meshes[rank];
    
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
       mesh.u0 = mesh.u;
       mesh.v0 = mesh.v;
       int max_outer_iterations=100;
           //simple算法迭代
  
        MPI_Barrier(MPI_COMM_WORLD);
        
    for(int n=1;n<=max_outer_iterations;n++) {
         MPI_Barrier(MPI_COMM_WORLD);
        //1离散动量方程 
        double l2_norm_x, l2_norm_y;
        
       
        movement_function(mesh,equ_u,equ_v,10000);
        equ_u.build_matrix();
        equ_v.build_matrix();


    
        //3求解线性方程组
        double epsilon_uv=0.01;
       
        MPI_Barrier(MPI_COMM_WORLD);
        VectorXd x_v(mesh.internumber),y_v(mesh.internumber);
        
        CG_parallel(equ_u,mesh,equ_u.source,x_v,1e-16,70,rank,num_procs,l2_norm_x);
        
        
        CG_parallel(equ_v,mesh,equ_v.source,y_v,1e-16,70,rank,num_procs,l2_norm_y);
        vectorToMatrix(x_v,mesh.u,mesh);
        vectorToMatrix(y_v,mesh.v,mesh);
        MPI_Barrier(MPI_COMM_WORLD);
       
        
        
        

      
        
         MPI_Barrier(MPI_COMM_WORLD);

       
        exchangeColumns(mesh.u, rank, num_procs);
        exchangeColumns(mesh.v, rank, num_procs);
        exchangeColumns(equ_u.A_p, rank, num_procs);
         MPI_Barrier(MPI_COMM_WORLD);
       
        //4速度插值到面
        face_velocity(mesh ,equ_u);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        double epsilon_p=1e-5;
       
        pressure_function(mesh, equ_p, equ_u);
       
        // 重新更新源项并重建矩阵
        equ_p.build_matrix();
        //求解
        VectorXd p_v(mesh.internumber);
        CG_parallel(equ_p,mesh,equ_p.source,p_v,1e-17,80,rank,num_procs,l2_norm_p);
        
        vectorToMatrix(p_v,mesh.p_prime,mesh);
         MPI_Barrier(MPI_COMM_WORLD);
       
        exchangeColumns(mesh.p_prime, rank, num_procs); 
        MPI_Barrier(MPI_COMM_WORLD);
        //8压力修正
        correct_pressure(mesh,equ_u);
        MPI_Barrier(MPI_COMM_WORLD);
        //9速度修正
        correct_velocity(mesh,equ_u);
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        //10更新压力
        mesh.p=mesh.p_star;
       
        MPI_Barrier(MPI_COMM_WORLD);
  

       
     
        MPI_Barrier(MPI_COMM_WORLD);
        exchangeColumns(mesh.p, rank, num_procs);
       
        MPI_Barrier(MPI_COMM_WORLD);
       
       
     // 计算全局残差
double global_l2_norm_x, global_l2_norm_y, global_l2_norm_p;
// 使用 MPI_Allreduce 而不是 MPI_Reduce 以便所有进程都能获得结果
MPI_Allreduce(&l2_norm_x, &global_l2_norm_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(&l2_norm_y, &global_l2_norm_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(&l2_norm_p, &global_l2_norm_p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

// 计算归一化残差
double norm_x = sqrt(global_l2_norm_x)/mesh.internumber;
double norm_y = sqrt(global_l2_norm_y)/mesh.internumber;
double norm_p = sqrt(global_l2_norm_p)/mesh.internumber;

// 只在主进程(rank=0)打印残差信息
if(rank == 0) {
    std::cout << scientific 
              << "时间步: " << i 
              << " 迭代轮数: " << n 
              << " u速度残差: " << setprecision(6) << norm_x
              << " v速度残差: " << setprecision(6) << norm_y
              << " 压力残差: " << setprecision(6) << norm_p
              << std::endl;
}

// 检查收敛性
int local_converged = (norm_x < 1e-8) && 
                      (norm_y < 1e-8) && 
                      (norm_p < 1e-10);

// 同步所有进程的收敛状态
int global_converged;
MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

if(global_converged) {
    if(rank == 0) {
        std::cout << "所有进程达到收敛条件" << std::endl;
    }
    break;
}
    MPI_Barrier(MPI_COMM_WORLD);
    }
    //显式时间推进
    mesh.u = mesh.u0 + dt*mesh.u_star;
    mesh.v = mesh.v0 + dt*mesh.v_star;
    if (i % 5 == 0) {
       saveMeshData(mesh,rank);
   
    
    }
    }
     
    // 最后显示实际计算总耗时
    auto total_elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time0).count();
    std::cout << "\n计算完成 总耗时: " << total_elapsed_time << "秒" << std::endl;
  
    saveMeshData(mesh,rank);
    MPI_Finalize();
    
     
    return 0;
}
   
    