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
        p_file << mesh.p;
        p_file.close();

        std::cout << "进程 " << rank << " 的数据已保存到文件" << std::endl;
    }
    catch(const std::exception& e) {
        std::cerr << "保存数据时出错: " << e.what() << std::endl;
    }
}


// 从解向量转换为场矩阵
void vectorToMatrix(const VectorXd& x, MatrixXd& phi, const Mesh& mesh) {
    for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                int n = mesh.interid(i, j) - 1; // 获取对应的解向量索引
                phi(i, j) = x[n];
            }
        }
    }
}

// 从场矩阵转换为解向量
void matrixToVector(const MatrixXd& phi, VectorXd& x, const Mesh& mesh) {
    for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                int n = mesh.interid(i, j) - 1; // 获取对应的解向量索引
                x[n] = phi(i, j);
            }
        }
    }
}
void Parallel_correction(Mesh mesh,Equation equ,MatrixXd &phi1,MatrixXd &phi2){
for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                if (mesh.bctype(i, j+1) == -3)
                {
                     phi1(i, j)-=equ.A_e(i, j)*phi2(i, j+1);
                }
                 if (mesh.bctype(i, j-1) == -3)
                {
                     phi1(i, j)-=equ.A_w(i, j)*phi2(i, j-1);
                }
                
               
            }
        }
    }
}
// 并行共轭梯度（CG）算法实现
// 输入：
// A - 系数矩阵（稀疏格式）
// b - 右端项向量
// x - 初始解向量，结果将存储在此
// epsilon - 收敛精度
// max_iter - 最大迭代次数
// rank - 当前进程的标识符（MPI）
// num_procs - 总进程数量（MPI）
void CG_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon, int max_iter, int rank, int num_procs,double& r0) {
    int n = equ.A.rows();

    SparseMatrix<double> A=equ.A;
    // 计算初始残差 r = b - A * x
    MatrixXd r_field(mesh.ny+2,mesh.nx+2) ,x_field(mesh.ny+2,mesh.nx+2);
    VectorXd r = b - A * x;
    vectorToMatrix(r,r_field,mesh);
    vectorToMatrix(x,x_field,mesh);
    exchangeColumns(x_field,rank,num_procs);
    Parallel_correction(mesh,equ,r_field,x_field);
    matrixToVector(r_field,r,mesh);
    // 初始化搜索方向 p = r
    VectorXd p = r;         
    // 存储矩阵与方向向量乘积 Ap
    VectorXd Ap(n);         

    // 计算初始残差范数 r_norm = ||r||^2
    double r_norm = r.squaredNorm();
    // 计算右端项范数 b_norm = ||b||^2，用于归一化停止条件
    double b_norm = b.squaredNorm();
    // 停止条件阈值 tol = epsilon^2 * ||b||^2
    double tol = epsilon * epsilon * b_norm;

   

    // 全局归约初始残差范数
    double global_r_norm;
    MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int iter = 0; // 当前迭代次数
        
    while (iter < max_iter && global_r_norm > tol) {
        // 第一步：计算矩阵-向量乘积 Ap = A * p
        // --- TODO: 在此处实现并行矩阵-向量乘积 ---
        Ap=A*p;
        MatrixXd p_field(mesh.ny+2,mesh.nx+2) ,Ap_field(mesh.ny+2,mesh.nx+2);
        vectorToMatrix(p,p_field,mesh);
        vectorToMatrix(Ap,Ap_field,mesh);
        exchangeColumns(p_field,rank,num_procs);
        Parallel_correction(mesh,equ,Ap_field,p_field);
        matrixToVector(Ap_field,Ap,mesh);


        // 第二步：计算 alpha = (r, r) / (p, Ap)，其中 (p, Ap) 是向量点积
        double local_dot_p_Ap = p.dot(Ap); // 计算本地点积
        double global_dot_p_Ap;
        MPI_Allreduce(&local_dot_p_Ap, &global_dot_p_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // 全局归约计算 (p, Ap)
        double alpha = global_r_norm / global_dot_p_Ap;

        // 第三步：更新解向量 x 和残差向量 r
        x += alpha * p; // 更新解向量
        r -= alpha * Ap; // 更新残差向量

        // 第四步：计算新的残差范数 ||r||^2
        double new_r_norm = r.squaredNorm(); // 局部范数
        double global_new_r_norm;
        MPI_Allreduce(&new_r_norm, &global_new_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // 全局归约计算范数

      

        // 第五步：计算 beta = (r_new, r_new) / (r_old, r_old)，并更新搜索方向向量 p
        double beta = global_new_r_norm / global_r_norm; // 比例因子
        p = r + beta * p; // 更新搜索方向

        // 更新旧的残差范数
        global_r_norm = global_new_r_norm;
        iter++;
        // 打印当前迭代信息（仅主进程）
        if (rank == 0) {
            cout << "Iteration " << iter << " Residual norm: " << sqrt(global_new_r_norm) << endl;
        }
        r0= global_r_norm;
    }

  
}
void CG_parallel2(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon, int max_iter, int rank, int num_procs,double& r0) {
    int n = equ.A.rows();
     MPI_Barrier(MPI_COMM_WORLD);
    SparseMatrix<double> A=equ.A;
    // 计算初始残差 r = b - A * x
    MatrixXd r_field(mesh.ny+2,mesh.nx+2) ,x_field(mesh.ny+2,mesh.nx+2);
    VectorXd r = b - A * x;
    /*vectorToMatrix(r,r_field,mesh);
    vectorToMatrix(x,x_field,mesh);
    MPI_Barrier(MPI_COMM_WORLD);
    exchangeColumns(x_field,rank,num_procs);
    MPI_Barrier(MPI_COMM_WORLD);
    Parallel_correction(mesh,equ,r_field,x_field);
    MPI_Barrier(MPI_COMM_WORLD);
    matrixToVector(r_field,r,mesh);*/
    // 初始化搜索方向 p = r
    VectorXd p = r;         
    // 存储矩阵与方向向量乘积 Ap
    VectorXd Ap(n);         

    // 计算初始残差范数 r_norm = ||r||^2
    double r_norm = r.squaredNorm();
    // 计算右端项范数 b_norm = ||b||^2，用于归一化停止条件
    double b_norm = b.squaredNorm();
    // 停止条件阈值 tol = epsilon^2 * ||b||^2
    double tol = epsilon * epsilon * b_norm;


    // 全局归约初始残差范数
    double global_r_norm;
    MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int iter = 0; // 当前迭代次数
    
    while (iter < max_iter ) {
        // 第一步：计算矩阵-向量乘积 Ap = A * p
        // --- TODO: 在此处实现并行矩阵-向量乘积 ---
        MPI_Barrier(MPI_COMM_WORLD);
        Ap=A*p;
        MatrixXd p_field(mesh.ny+2,mesh.nx+2) ,Ap_field(mesh.ny+2,mesh.nx+2);
        vectorToMatrix(p,p_field,mesh);
        vectorToMatrix(Ap,Ap_field,mesh);
        MPI_Barrier(MPI_COMM_WORLD);
        exchangeColumns(p_field,rank,num_procs);
        MPI_Barrier(MPI_COMM_WORLD);
        Parallel_correction(mesh,equ,Ap_field,p_field);
        MPI_Barrier(MPI_COMM_WORLD);
        matrixToVector(Ap_field,Ap,mesh);
        MPI_Barrier(MPI_COMM_WORLD);

        // 第二步：计算 alpha = (r, r) / (p, Ap)，其中 (p, Ap) 是向量点积
        double local_dot_p_Ap = p.dot(Ap); // 计算本地点积
        double global_dot_p_Ap;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&local_dot_p_Ap, &global_dot_p_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // 全局归约计算 (p, Ap)
        double alpha = global_r_norm / global_dot_p_Ap;

        // 第三步：更新解向量 x 和残差向量 r
        x += alpha * p; // 更新解向量
        r -= alpha * Ap; // 更新残差向量

        // 第四步：计算新的残差范数 ||r||^2
        double new_r_norm = r.squaredNorm(); // 局部范数
        double global_new_r_norm;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&new_r_norm, &global_new_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // 全局归约计算范数

        

        // 第五步：计算 beta = (r_new, r_new) / (r_old, r_old)，并更新搜索方向向量 p
        double beta = global_new_r_norm / global_r_norm; // 比例因子
        p = r + beta * p; // 更新搜索方向

        // 更新旧的残差范数
        global_r_norm = global_new_r_norm;
        iter++;

        // 打印当前迭代信息（仅主进程）
        if (rank == 0) {
            cout << "Iteration " << iter << " Residual norm: " << sqrt(global_new_r_norm) << endl;
        }
        r0= global_r_norm;
    }
      
    // 如果达到最大迭代次数但未收敛，发出警告
    if (rank == 0 && iter == max_iter) {
        cerr << "Warning: CG did not converge within the maximum number of iterations." << endl;
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
   auto start_time0 = chrono::steady_clock::now();  // 开始计时
  
       int max_outer_iterations=200;
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
        CG_parallel2(equ_u,mesh,equ_u.source,x_v,0.1,20,rank,num_procs,l2_norm_x);
        MPI_Barrier(MPI_COMM_WORLD);
        
        CG_parallel2(equ_v,mesh,equ_v.source,y_v,0.1,20,rank,num_procs,l2_norm_y);
        vectorToMatrix(x_v,mesh.u,mesh);
        vectorToMatrix(y_v,mesh.v,mesh);
        MPI_Barrier(MPI_COMM_WORLD);
       
        
        
        

      
        
         MPI_Barrier(MPI_COMM_WORLD);

       
        exchangeColumns(mesh.u, rank, num_procs);
        exchangeColumns(mesh.v, rank, num_procs);
        exchangeColumns(equ_u.A_p, rank, num_procs);
        
       
        //4速度插值到面
        face_velocity(mesh ,equ_u);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        double epsilon_p=1e-5;
        double global_l2_norm_p;
        pressure_function(mesh, equ_p, equ_u);
       
        // 重新更新源项并重建矩阵
        equ_p.build_matrix();
        //求解
        VectorXd p_v(mesh.internumber);
        CG_parallel2(equ_p,mesh,equ_p.source,p_v,1e-4,11,rank,num_procs,l2_norm_p);
        cout<<"完成计算"<<endl;
        vectorToMatrix(p_v,mesh.p_prime,mesh);

        

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
       
        
        
        //收敛性判断
    std::cout << scientific 
            << "进程 " << rank 
             
        << " 子循环轮数 " << n 
           
            << " u速度残差 " << setprecision(6) << (l2_norm_x/mesh.internumber)
            << " v速度残差 " << setprecision(6) << (l2_norm_y/mesh.internumber)
            << " 压力残差 " <<  setprecision(6) << (l2_norm_p/mesh.internumber)
            << "\n" <<  endl;
            
       /*if((global_l2_norm_x/mesh.internumber) < 1.5*1e-5 && (global_l2_norm_y/mesh.internumber) < 1e-5 &&(global_l2_norm_p/mesh.internumber) < 1e-7) { 
            std::cout << "线程 " << rank << " 达到收敛条件" << std::endl;
            
            break;
        
         
        }*/
   
    MPI_Barrier(MPI_COMM_WORLD);
    }
    
   
    // 最后显示实际计算总耗时
    auto total_elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time0).count();
    std::cout << "\n计算完成 总耗时: " << total_elapsed_time << "秒" << std::endl;
  
    saveMeshData(mesh,rank);
    MPI_Finalize();
    
     
    return 0;
}
   
    

