#include "DNS.h"
#include <filesystem>
#include <chrono>
#include "parallel.h"
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

        std::cout << "线程 " << rank << " 的数据已保存到文件" << std::endl;
    }
    catch(const std::exception& e) {
        std::cerr << "保存数据时出错: " << e.what() << std::endl;
    }
}
// 实现在 DNS.cpp 中
void printUfaceColumnSums(const Mesh& mesh, int rank, int num_procs) {
    double sum_second = 0.0;
    double sum_secondlast = 0.0;
    
    // 计算第二列的和
    for(int i = 1; i <= mesh.ny ; i++) {
        if (mesh.u(i, 2)>=0)
        {
             sum_second += mesh.u(i, 2);
        }
        
        
    }
    
    // 计算倒数第二列的和
    for(int i = 1; i <= mesh.ny ; i++) {
        if ( mesh.u(i, mesh.nx-1 )>=0)
        {
            sum_secondlast += mesh.u(i, mesh.nx-1 );
        }
        
        
    }
    
    // 根据进程号打印结果
    if(rank == 0) {
        std::cout << "线程 0 u_face倒数第二列之和: " 
                  << sum_secondlast << std::endl;
    }
    else if(rank == num_procs - 1) {
        std::cout << "线程 " << rank << " u_face第二列之和: " 
                  << sum_second << std::endl;
    }
    else {
        std::cout << "线程 " << rank 
                  << " u_face第二列之和: " << sum_second 
                  << " 倒数第二列之和: " << sum_secondlast 
                  << std::endl;
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
    // 循环执行
    for (int i = 0; i <= timesteps; ++i) { 
        
        cout<<"时间步长 "<< i <<std::endl;
        
       
       //记录上一个时间步长的u v
       mesh.u0 = mesh.u;
       mesh.v0 = mesh.v;
       //每步最大迭代次数
       int max_outer_iterations=100;
           //simple算法迭代
 
        MPI_Barrier(MPI_COMM_WORLD);
        
    for(int n=1;n<=max_outer_iterations;n++) {
         MPI_Barrier(MPI_COMM_WORLD);
        //1离散动量方程 
        double global_l2_norm_x, global_l2_norm_y;
        movement_function(mesh,equ_u,equ_v,10000);
         
        
        equ_u.build_matrix();
        equ_v.build_matrix();
        //3求解线性方程组
        double epsilon_uv=0.01;
        
         for (int iter = 0; iter < 1; ++iter) {
        // 1. 求解子域内的 u 和 v
        
        solve(equ_u, epsilon_uv, l2_norm_x, mesh.u); // 子域u的解
        solve(equ_v, epsilon_uv, l2_norm_y, mesh.v); // 子域v的解

        // 2. 同步各子域：交换重叠区域的值
        MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程同步
        exchangeColumns(mesh.u, rank, num_procs);   // 交换 u 的边界值
        exchangeColumns(mesh.v, rank, num_procs);   // 交换 v 的边界值
        MPI_Barrier(MPI_COMM_WORLD);
        movement_function(mesh,equ_u,equ_v,10000);
      
        equ_u.build_matrix();
        equ_v.build_matrix();
        //printMatrix(mesh.u,"u",4);
        // 3. 汇总全局残差
        
        MPI_Allreduce(&l2_norm_x, &global_l2_norm_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&l2_norm_y, &global_l2_norm_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 计算全局残差
        global_l2_norm_x = sqrt(global_l2_norm_x);
        global_l2_norm_y = sqrt(global_l2_norm_y);
        if (rank == 0) {
            std::cout << "Iteration " << iter << ": "
                      << "Global l2 norm for u = " << global_l2_norm_x 
                      << ", v = " << global_l2_norm_y << std::endl;
        }
         }

       
        MPI_Barrier(MPI_COMM_WORLD);
        
        exchangeColumns(equ_u.A_p, rank, num_procs);
        exchangeColumns(equ_v.A_p, rank, num_procs);
        MPI_Barrier(MPI_COMM_WORLD);
        //4速度插值到面
        face_velocity(mesh ,equ_u);
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        double epsilon_p=1e-5;
        double global_l2_norm_p;
        // 初始化压力方程求解的迭代过程
    for (int iter2 = 0; iter2 < 3; ++iter2) {
        // 1. 求解子域内的 p（只求解压力方程）
        
        solve(equ_p, epsilon_p, l2_norm_p, mesh.p_prime); // 对压力方程求解
        MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程同步
        exchangeColumns(mesh.p_prime, rank, num_procs);   // 交换压力重叠区域
        MPI_Barrier(MPI_COMM_WORLD);
        pressure_function(mesh,equ_p,equ_u);
        equ_p.build_matrix();
        // 2. 汇总全局残差
        
        MPI_Allreduce(&l2_norm_p, &global_l2_norm_p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 计算全局残差
        global_l2_norm_p = sqrt(global_l2_norm_p);

        if (rank == 0) {
            std::cout << "Iteration " << iter2 << ": "
                      << "Global l2 norm for pressure = " << global_l2_norm_p << std::endl;
        }
        
        
    }
         
         
         
         MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程同步
        exchangeColumns(mesh.p_prime, rank, num_procs); 
    
        //8压力修正
        correct_pressure(mesh,equ_u);
        
        //9速度修正
        correct_velocity(mesh,equ_u);
       
        
        
        //10更新压力
        mesh.p=mesh.p_star;
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        exchangeColumns(mesh.u, rank, num_procs);
    exchangeColumns(mesh.v, rank, num_procs);
    exchangeColumns(mesh.u0, rank, num_procs);
    exchangeColumns(mesh.v0, rank, num_procs);
    exchangeColumns(mesh.u_face, rank, num_procs);
    exchangeColumns(mesh.v_face, rank, num_procs);
    exchangeColumns(mesh.u_star, rank, num_procs);
    exchangeColumns(mesh.v_star, rank, num_procs);
    exchangeColumns(mesh.p, rank, num_procs);
    exchangeColumns(mesh.p_prime, rank, num_procs);
    exchangeColumns(mesh.p_star, rank, num_procs);
       
        
        MPI_Barrier(MPI_COMM_WORLD);
        //收敛性判断
        std::cout << scientific 
            << "线程 " << rank 
            << " 时间步 " << i
            << " 子域循环轮数 " << n 
         
            << " u速度残差 " << setprecision(6) << (global_l2_norm_x/mesh.internumber)
            << " v速度残差 " << setprecision(6) << (global_l2_norm_y/mesh.internumber)
            << " 压力残差 " <<  setprecision(6) << (global_l2_norm_p/mesh.internumber)
            << "\n" <<  endl;
            
       /*if((l2x/mesh.internumber) < 1e-8 & (l2y/mesh.internumber) < 1e-8 & (l2p/mesh.internumber) < 1e-8) { 
            std::cout << "线程 " << rank << " 达到收敛条件" << std::endl;
            
            break;
        
         
        }*/
  
    }
    
    /*//显式时间推进
    mesh.u = mesh.u0 + dt*mesh.u;
    mesh.v = mesh.v0 + dt*mesh.v;
    exchangeColumns(mesh.u, rank, num_procs);
    exchangeColumns(mesh.v, rank, num_procs);
    exchangeColumns(mesh.u0, rank, num_procs);
    exchangeColumns(mesh.v0, rank, num_procs);
    MPI_Barrier(MPI_COMM_WORLD);
    sub_meshes[rank]=mesh;*/
    
    }
    
    saveMeshData(mesh,rank);
    MPI_Finalize();
    
    
    return 0;
}
   
    

