#include "parallel.h"
#include <omp.h>

// 定义全局变量
double total_comm_time = 0.0; // 通信时间
int total_comm_count = 0;     // 通信次数
double start_time, end_time;
int totalcount = 0;  
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
// 发送矩阵列数据(非阻塞)
/*void sendMatrixColumn(const MatrixXd& src_matrix, int src_col, 
                     MatrixXd& dst_matrix, int dst_col, 
                     int target_rank, int tag) {
    int rows = src_matrix.rows();
    
    // 创建发送缓冲区
    std::vector<double> send_buffer(rows);
    
    // 将矩阵列复制到缓冲区
    for(int i = 0; i < rows; i++) {
        send_buffer[i] = src_matrix(i, src_col);
    }
    
    // 非阻塞发送数据
    MPI_Request request;
    MPI_Isend(send_buffer.data(), rows, MPI_DOUBLE, target_rank, tag, 
              MPI_COMM_WORLD, &request);
    
    // 等待发送完成
    MPI_Status status;
    MPI_Wait(&request, &status);
}

// 接收矩阵列数据(非阻塞)
void recvMatrixColumn(MatrixXd& dst_matrix, int dst_col,
                     int src_rank, int tag) {
    int rows = dst_matrix.rows();
    
    // 创建接收缓冲区
    std::vector<double> recv_buffer(rows);
    
    // 非阻塞接收数据
    MPI_Request request;
    MPI_Irecv(recv_buffer.data(), rows, MPI_DOUBLE, src_rank, tag, 
              MPI_COMM_WORLD, &request);
    
    // 等待接收完成
    MPI_Status status;
    MPI_Wait(&request, &status);
    
    // 将接收的数据复制到目标矩阵列
    for(int i = 0; i < rows; i++) {
        dst_matrix(i, dst_col) = recv_buffer[i];
    }
}*/

/*void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
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
}*/
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    std::vector<MPI_Request> requests;
    std::vector<std::vector<double>> send_buffers;
    std::vector<std::vector<double>> recv_buffers;
    const int rows = matrix.rows();

    // 预分配缓冲区
    if(rank == 0) {
        // 0号进程需要2个发送和2个接收缓冲区
        send_buffers.resize(2, std::vector<double>(rows));
        recv_buffers.resize(2, std::vector<double>(rows));
        requests.resize(4);

        // 准备发送数据
        for(int i = 0; i < rows; i++) {
            send_buffers[0][i] = matrix(i, matrix.cols()-4);
            send_buffers[1][i] = matrix(i, matrix.cols()-3);
        }

        // 非阻塞发送
        MPI_Isend(send_buffers[0].data(), rows, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(send_buffers[1].data(), rows, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &requests[1]);

        // 非阻塞接收
        MPI_Irecv(recv_buffers[0].data(), rows, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(recv_buffers[1].data(), rows, MPI_DOUBLE, rank+1, 3, MPI_COMM_WORLD, &requests[3]);
    }
    else if(rank == num_procs-1) {
        // 最后一个进程需要2个发送和2个接收缓冲区
        send_buffers.resize(2, std::vector<double>(rows));
        recv_buffers.resize(2, std::vector<double>(rows));
        requests.resize(4);

        // 准备发送数据
        for(int i = 0; i < rows; i++) {
            send_buffers[0][i] = matrix(i, 2);
            send_buffers[1][i] = matrix(i, 3);
        }

        // 非阻塞发送
        MPI_Isend(send_buffers[0].data(), rows, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(send_buffers[1].data(), rows, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, &requests[1]);

        // 非阻塞接收
        MPI_Irecv(recv_buffers[0].data(), rows, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(recv_buffers[1].data(), rows, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &requests[3]);
    }
    else {
        // 中间进程需要4个发送和4个接收缓冲区
        send_buffers.resize(4, std::vector<double>(rows));
        recv_buffers.resize(4, std::vector<double>(rows));
        requests.resize(8);

        // 准备发送数据
        for(int i = 0; i < rows; i++) {
            send_buffers[0][i] = matrix(i, 2);
            send_buffers[1][i] = matrix(i, 3);
            send_buffers[2][i] = matrix(i, matrix.cols()-4);
            send_buffers[3][i] = matrix(i, matrix.cols()-3);
        }

        // 非阻塞发送到左右两侧
        MPI_Isend(send_buffers[0].data(), rows, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(send_buffers[1].data(), rows, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(send_buffers[2].data(), rows, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(send_buffers[3].data(), rows, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &requests[3]);

        // 非阻塞接收来自左右两侧的数据
        MPI_Irecv(recv_buffers[0].data(), rows, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[4]);
        MPI_Irecv(recv_buffers[1].data(), rows, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &requests[5]);
        MPI_Irecv(recv_buffers[2].data(), rows, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &requests[6]);
        MPI_Irecv(recv_buffers[3].data(), rows, MPI_DOUBLE, rank+1, 3, MPI_COMM_WORLD, &requests[7]);
    }

    // 等待所有通信完成
    std::vector<MPI_Status> statuses(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());

    // 更新矩阵数据
    if(rank == 0) {
        for(int i = 0; i < rows; i++) {
            matrix(i, matrix.cols()-2) = recv_buffers[0][i];
            matrix(i, matrix.cols()-1) = recv_buffers[1][i];
        }
    }
    else if(rank == num_procs-1) {
        for(int i = 0; i < rows; i++) {
            matrix(i, 0) = recv_buffers[0][i];
            matrix(i, 1) = recv_buffers[1][i];
        }
    }
    else {
        for(int i = 0; i < rows; i++) {
            matrix(i, 0) = recv_buffers[0][i];
            matrix(i, 1) = recv_buffers[1][i];
            matrix(i, matrix.cols()-2) = recv_buffers[2][i];
            matrix(i, matrix.cols()-1) = recv_buffers[3][i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
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
void Parallel_correction2(Mesh mesh,Equation equ,MatrixXd &phi1,MatrixXd &phi2){
for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                if (mesh.bctype(i, j+1) == -3)
                {
                     phi1(i, j)+=equ.A_e(i, j)*phi2(i, j+1);
                }
                 if (mesh.bctype(i, j-1) == -3)
                {
                     phi1(i, j)+=equ.A_w(i, j)*phi2(i, j-1);
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

void CG_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon, 
                int max_iter, int rank, int num_procs, double& r0) {
    int n = equ.A.rows();
    SparseMatrix<double> A = equ.A;
     
    // 计算初始残差
    VectorXd r = b - A * x;//矩阵向量乘 n1
    MPI_Barrier(MPI_COMM_WORLD);
    MatrixXd r_field(mesh.ny+2, mesh.nx+2), x_field(mesh.ny+2, mesh.nx+2);
    //交换矩阵重叠区域并计算
    vectorToMatrix(r, r_field, mesh);
    vectorToMatrix(x, x_field, mesh);
    MPI_Barrier(MPI_COMM_WORLD);
    exchangeColumns(x_field, rank, num_procs);
    MPI_Barrier(MPI_COMM_WORLD);
    //修正重叠单元
    Parallel_correction2(mesh, equ, r_field, x_field);
    MPI_Barrier(MPI_COMM_WORLD);
    //写回向量
    matrixToVector(r_field, r, mesh);
    MPI_Barrier(MPI_COMM_WORLD);
    VectorXd p = r;         
    VectorXd Ap(n);
    
    // 计算初始残差和基准残差
    double r_norm = r.squaredNorm();
    double b_norm = b.squaredNorm();
    
    double local_b_norm = b_norm;
    double global_b_norm;
    // 使用 MPI_Allreduce 来将各个进程的 b_norm 求最大值（或总和等），确保每个进程能够看到全局的 b_norm
MPI_Allreduce(&local_b_norm, &global_b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

// 检查全局的 b_norm
if (global_b_norm < 1e-13) {
    x.setZero();
    r0 = 0.0;
    if (rank==0)
    {
       cout<<"全局b_norm小于1e-13，直接返回"<<endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    return;  // 一旦判断为小于阈值，直接在所有进程处执行return
}
    // 使用绝对残差判据
    double tol = epsilon * epsilon; // 直接使用给定的epsilon作为绝对收敛判据
    
    double global_r_norm;
    //全局规约残差
    MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     r0 = sqrt(global_r_norm);  // 更新初始r0
    int iter = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    while (iter < max_iter) {
        // 计算 Ap
        Ap = A * p;
        MatrixXd p_field(mesh.ny+2, mesh.nx+2), Ap_field(mesh.ny+2, mesh.nx+2);
        vectorToMatrix(p, p_field, mesh);
        vectorToMatrix(Ap, Ap_field, mesh);
        MPI_Barrier(MPI_COMM_WORLD);
        exchangeColumns(p_field, rank, num_procs);
        MPI_Barrier(MPI_COMM_WORLD);
        Parallel_correction(mesh, equ, Ap_field, p_field);
        MPI_Barrier(MPI_COMM_WORLD);
        matrixToVector(Ap_field, Ap, mesh);
        MPI_Barrier(MPI_COMM_WORLD);
        // 计算步长
        double local_dot_p_Ap = p.dot(Ap);
        double global_dot_p_Ap;
        MPI_Allreduce(&local_dot_p_Ap, &global_dot_p_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = global_r_norm / global_dot_p_Ap;
        MPI_Barrier(MPI_COMM_WORLD);
        // 更新解和残差
        x += alpha * p;
        r -= alpha * Ap;

        // 计算新残差范数
        double new_r_norm = r.squaredNorm();
        double global_new_r_norm;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&new_r_norm, &global_new_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // 更新r0为当前全局残差
        MPI_Barrier(MPI_COMM_WORLD);
        // 使用绝对残差判断收敛性
       

        // 更新搜索方向
        double beta = global_new_r_norm / global_r_norm;
        p = r + beta * p;
        global_r_norm = global_new_r_norm;

        // 保存当前残差
        r0 = sqrt(global_r_norm);

        /*if(rank == 0 && iter % 5 == 0) {
            std::cout << "Iteration " << iter 
                     << " Absolute residual: " << sqrt(global_new_r_norm) 
                     << std::endl;
        }*/

        iter++;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // 确保最终r0同步
   // MPI_Bcast(&r0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}


void BiCGSTAB_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon,
    int max_iter, int rank, int num_procs, double& r0) {
    
int n = equ.A.rows();
SparseMatrix<double> A = equ.A;

VectorXd r = b - A * x;
MatrixXd r_field(mesh.ny+2, mesh.nx+2), x_field(mesh.ny+2, mesh.nx+2);
vectorToMatrix(r, r_field, mesh);
vectorToMatrix(x, x_field, mesh);
exchangeColumns(x_field, rank, num_procs);
Parallel_correction2(mesh, equ, r_field, x_field);
matrixToVector(r_field, r, mesh);

    VectorXd r_tld = r;
    VectorXd p = r, v = VectorXd::Zero(n), s, t;

double rho_old = 1.0, alpha = 1.0, omega = 1.0;
double rho_new, beta;

double r_norm = r.squaredNorm();
double local_b_norm = b.squaredNorm(), global_b_norm;
MPI_Allreduce(&local_b_norm, &global_b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

if (global_b_norm < 1e-13) {
x.setZero();
r0 = 0.0;
MPI_Barrier(MPI_COMM_WORLD);
return;
}

double global_r_norm;
MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
r0 = sqrt(global_r_norm);

int iter = 0;
while (iter < max_iter) {
rho_new = r_tld.dot(r);
        

if (iter == 0) {
p = r;
} else {
beta = (rho_new / rho_old) * (alpha / omega);
p = r + beta * (p - omega * v);
}

v = A * p;
MatrixXd p_field(mesh.ny+2, mesh.nx+2), v_field(mesh.ny+2, mesh.nx+2);
vectorToMatrix(p, p_field, mesh);
vectorToMatrix(v, v_field, mesh);
exchangeColumns(p_field, rank, num_procs);
Parallel_correction(mesh, equ, v_field, p_field);
matrixToVector(v_field, v, mesh);

double local_dot = r_tld.dot(v), global_dot;
MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
alpha = rho_new / global_dot;

// s = r - α*v
s = r - alpha * v;

// s 范数（不再提前退出）
double s_norm = s.squaredNorm(), global_s_norm;
MPI_Allreduce(&s_norm, &global_s_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // if (sqrt(global_s_norm) < epsilon) {
        //     x += alpha * p;
        //     break;
        // }

t = A * s;
MatrixXd s_field(mesh.ny+2, mesh.nx+2), t_field(mesh.ny+2, mesh.nx+2);
vectorToMatrix(s, s_field, mesh);
vectorToMatrix(t, t_field, mesh);
exchangeColumns(s_field, rank, num_procs);
Parallel_correction(mesh, equ, t_field, s_field);
matrixToVector(t_field, t, mesh);

double ts_dot = t.dot(s);
double tt_dot = t.dot(t);
double global_ts_dot, global_tt_dot;
MPI_Allreduce(&ts_dot, &global_ts_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(&tt_dot, &global_tt_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
omega = global_ts_dot / global_tt_dot;

x += alpha * p + omega * s;
r = s - omega * t;

r_norm = r.squaredNorm();
MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
r0 = sqrt(global_r_norm);

        // 不再使用残差判断终止
        // if (r0 < epsilon) break;

rho_old = rho_new;
iter++;
    }

MPI_Barrier(MPI_COMM_WORLD);
}