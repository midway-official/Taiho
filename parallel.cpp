#include "parallel.h"


// 发送矩阵列数据
void sendMatrixColumn(const MatrixXd& src_matrix, int src_col, 
                     MatrixXd& dst_matrix, int dst_col, 
                     int target_rank, int tag) {
    int rows = src_matrix.rows();
    
    // 创建发送缓冲区
    std::vector<double> send_buffer(rows);
    
    // 将矩阵列复制到缓冲区
    for(int i = 0; i < rows; i++) {
        send_buffer[i] = src_matrix(i, src_col);
    }
    
    // 发送数据
    MPI_Send(send_buffer.data(), rows, MPI_DOUBLE, target_rank, tag, MPI_COMM_WORLD);
}

// 接收矩阵列数据
void recvMatrixColumn(MatrixXd& dst_matrix, int dst_col,
                     int src_rank, int tag) {
    int rows = dst_matrix.rows();
    
    // 创建接收缓冲区
    std::vector<double> recv_buffer(rows);
    
    // 接收数据
    MPI_Status status;
    MPI_Recv(recv_buffer.data(), rows, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &status);
    
    // 将接收的数据复制到目标矩阵列
    for(int i = 0; i < rows; i++) {
        dst_matrix(i, dst_col) = recv_buffer[i];
    }
}

void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    // 特殊处理0号线程
    if(rank == 0) {
        // 只发送给右侧线程
        // 发送倒数第4、3列到右侧线程的第1、2列
        sendMatrixColumn(matrix, matrix.cols()-4, matrix, 0, rank+1);
        sendMatrixColumn(matrix, matrix.cols()-3, matrix, 1, rank+1);
        
        // 接收右侧线程发来的列
        recvMatrixColumn(matrix, matrix.cols()-2, rank+1);
        recvMatrixColumn(matrix, matrix.cols()-1, rank+1);
    }
    // 特殊处理最后一个线程
    else if(rank == num_procs-1) {
        // 只发送给左侧线程
        // 发送第3、4列到左侧线程的倒数第2、1列
        sendMatrixColumn(matrix, 2, matrix, matrix.cols()-2, rank-1);
        sendMatrixColumn(matrix, 3, matrix, matrix.cols()-1, rank-1);
        
        // 接收左侧线程发来的列
        recvMatrixColumn(matrix, 0, rank-1);
        recvMatrixColumn(matrix, 1, rank-1);
    }
    // 处理中间线程
    else {
        // 先发送给左侧线程
        sendMatrixColumn(matrix, 2, matrix, matrix.cols()-2, rank-1);
        sendMatrixColumn(matrix, 3, matrix, matrix.cols()-1, rank-1);
        
        // 再发送给右侧线程
        sendMatrixColumn(matrix, matrix.cols()-4, matrix, 0, rank+1);
        sendMatrixColumn(matrix, matrix.cols()-3, matrix, 1, rank+1);
        
        // 接收来自左侧线程的数据
        recvMatrixColumn(matrix, 0, rank-1);
        recvMatrixColumn(matrix, 1, rank-1);
        
        // 接收来自右侧线程的数据
        recvMatrixColumn(matrix, matrix.cols()-2, rank+1);
        recvMatrixColumn(matrix, matrix.cols()-1, rank+1);
    }
    
    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
}
void gsstep(Equation& equation,  double& l2_norm, Eigen::MatrixXd& phi) {
    // 创建解向量，长度为内部点数量
    Eigen::VectorXd x(equation.mesh.internumber);

    // 根据 interid 构建初始解向量，遍历整个网格
    for (int i = 0; i <= equation.n_y + 1; i++) {
        for (int j = 0; j <= equation.n_x + 1; j++) {
            if (equation.mesh.bctype(i, j) == 0) {  // 确认是内部点
                int n = equation.mesh.interid(i, j) - 1;
                if (n >= 0 && n < equation.mesh.internumber) {
                    x[n] = phi(i, j);
                }
            }
        }
    }

    // 计算残差并初始化
    l2_norm = (equation.A * x - equation.source).norm();

    // 进行一次高斯赛德尔迭代
    Eigen::VectorXd newX = x;
    for (int i = 0; i < equation.mesh.internumber; ++i) {
        double sigma = 0.0;

        // 计算当前点的 sigma（非对角项的累加和）
        for (Eigen::SparseMatrix<double>::InnerIterator it(equation.A, i); it; ++it) {
            int j = it.col();
            if (j != i) {
                sigma += it.value() * newX[j];  // 使用已更新的解值
            }
        }

        // 更新当前点的解
        newX[i] = (equation.source[i] - sigma) / equation.A.coeff(i, i);
    }

    // 将结果写回网格，同样遍历整个网格
    for (int i = 0; i <= equation.n_y + 1; i++) {
        for (int j = 0; j <= equation.n_x + 1; j++) {
            if (equation.mesh.bctype(i, j) == 0) {  // 确认是内部点
                int n = equation.mesh.interid(i, j) - 1;
                if (n >= 0 && n < equation.mesh.internumber) {
                    phi(i, j) = newX[n];  // 更新网格中的解
                }
            }
        }
    }

    
}

void parallelGs_u(Equation& equ_u0, Equation& equ_v0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter) {
    int iter = 0;
    bool converged = false;

    while (iter < max_iter ) {
        // 第一次调用 gsstep 执行高斯赛德尔迭代
        gsstep(equ_u0, l2_norm_x, mesh.u);

        // 使用交换列操作在不同进程之间交换边界数据
        exchangeColumns(mesh.u, rank, num_procs);

        // 计算 l2_norm_x，检查是否收敛
        double local_l2_norm = 0.0;
        MPI_Allreduce(&l2_norm_x, &local_l2_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        l2_norm_x = sqrt(local_l2_norm);  // 汇总所有进程的 l2_norm_x 值并计算最终的结果

    
        // 如果尚未收敛，则更新 source 并进行下一步的计算
        movement_function(mesh, equ_u0, equ_v0, 10000);

        // 重新更新源项并重建矩阵
        equ_u0.build_matrix();
          

        iter++;
    }

    
}
void parallelGs_v(Equation& equ_v0, Equation& equ_u0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter) {
    int iter = 0;
    bool converged = false;

    while (iter < max_iter ) {
        // 第一次调用 gsstep 执行高斯赛德尔迭代
        gsstep(equ_v0, l2_norm_x, mesh.v);

        // 使用交换列操作在不同进程之间交换边界数据
        exchangeColumns(mesh.v, rank, num_procs);

        // 计算 l2_norm_x，检查是否收敛
        double local_l2_norm = 0.0;
        MPI_Allreduce(&l2_norm_x, &local_l2_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        l2_norm_x = sqrt(local_l2_norm);  // 汇总所有进程的 l2_norm_x 值并计算最终的结果

       
            // 如果尚未收敛，则更新 source 并进行下一步的计算
            movement_function(mesh, equ_u0, equ_v0, 10000);

            // 重新更新源项并重建矩阵
            equ_v0.build_matrix();
            // 在这里重新更新方程的源项（比如 A 和 b）
        

        iter++;
    }
}

void parallelGs_p(Equation& equ_p0, Equation& equ_u0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter) {
    int iter = 0;
    bool converged = false;

    while (iter < max_iter) {
        // 第一次调用 gsstep 执行高斯赛德尔迭代
        gsstep(equ_p0, l2_norm_x, mesh.p_prime);

        // 使用交换列操作在不同进程之间交换边界数据
        exchangeColumns(mesh.p_prime, rank, num_procs);

        // 计算 l2_norm_x，检查是否收敛
        double local_l2_norm = 0.0;
        MPI_Allreduce(&l2_norm_x, &local_l2_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        l2_norm_x = sqrt(local_l2_norm);  // 汇总所有进程的 l2_norm_x 值并计算最终的结果

       
            // 如果尚未收敛，则更新 source 并进行下一步的计算
            pressure_function(mesh, equ_p0, equ_u0);

            // 重新更新源项并重建矩阵
            equ_p0.build_matrix();
            // 在这里重新更新方程的源项（比如 A 和 b）
        

        iter++;
    }
    
}
