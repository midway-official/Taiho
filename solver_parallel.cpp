#include "DNS.h"
#include <filesystem>
#include <chrono>
#include "parallel.h"
namespace fs = std::filesystem;
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
 for (size_t i2 = 0; i2 < 2; i2++)
       {
        MPI_Barrier(MPI_COMM_WORLD);
    for(int n=1;n<=max_outer_iterations;n++) {
       
       
       
       
       
       
         MPI_Barrier(MPI_COMM_WORLD);
        //1离散动量方程 
        movement_function(mesh,equ_u,equ_v,100);

        //2组合线性方程组
    equ_u.build_matrix();
    equ_v.build_matrix();

        //3求解线性方程组
        double epsilon_uv=0.1;
        
       
    solve(equ_u, epsilon_uv, l2x, mesh.u);
    solve(equ_v, epsilon_uv, l2y, mesh.v);
       
       
        
        
        
        
       
        //4速度插值到面
        face_velocity(mesh ,equ_u);
        

        //5离散压力修正方程

        pressure_function(mesh,equ_p,equ_u);

       
        //6组合线性方程组
    equ_p.build_matrix();

        //7求解压力修正方程

       
        double epsilon_p=1e-5;
    solve(equ_p, epsilon_p, l2p, mesh.p_prime);
      
         
        
        //8压力修正
        correct_pressure(mesh,equ_u);
        
        //9速度修正
        correct_velocity(mesh,equ_u);
        
        
        
        
       
        //10更新压力
        mesh.p=mesh.p_star;
        
        
        
        //收敛性判断
        std::cout << scientific 
            << "线程 " << rank 
            << " 时间步 " << i
            << " 子域循环轮数 " << n 
            << " 全局循环轮数 " << i2
            << " u速度残差 " << setprecision(6) << (l2x/mesh.internumber)
            << " v速度残差 " << setprecision(6) << (l2y/mesh.internumber)
            << " 压力残差 " <<  setprecision(6) << (l2p/mesh.internumber)
            << "\n" <<  endl;
            
       /*if((l2x/mesh.internumber) < 1e-8 & (l2y/mesh.internumber) < 1e-8 & (l2p/mesh.internumber) < 1e-8) { 
            std::cout << "线程 " << rank << " 达到收敛条件" << std::endl;
            
            break;
        
         
        }*/

    MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
exchangeColumns(mesh.u, rank, num_procs);
exchangeColumns(mesh.v, rank, num_procs);
exchangeColumns(mesh.u_star, rank, num_procs);
exchangeColumns(mesh.v_star, rank, num_procs);

exchangeColumns(mesh.p, rank, num_procs);
    MPI_Barrier(MPI_COMM_WORLD);
    }


     MPI_Barrier(MPI_COMM_WORLD);
    //显式时间推进
    mesh.u = mesh.u0 + dt*mesh.u;
    mesh.v = mesh.v0 + dt*mesh.v;
    MPI_Barrier(MPI_COMM_WORLD);
    sub_meshes[rank]=mesh;
    
    }
    
    
    MPI_Finalize();
    
    
    return 0;
}
   
    

