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