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
