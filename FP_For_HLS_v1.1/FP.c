//
// Created by admin on 28.02.2022.
//

#define N 12
#define M 25

typedef struct FP_struct{
    int P;
    int I;
} FP_out;

FP_out nn_opt(int x1, int x2, int x3, int x4, int W1_tr[4][N], int W2_tr[N][N], int Wout_tr[N][2],
              int b1_tr[N], int b2_tr[N], int bout_tr[2], short tabl_sigma[1024]) {
    FP_out y_pred;
    int i, j, a1[N], a2[N];
        int in[4] = {x1, x2, x3, x4};
        // Matrix-vector multiplication and getting over threshold operation
        for (i = 0; i < N; i++) {
            long z1 = -b1_tr[i];
            for (j = 0; j < 4; j++) z1 += ((long long) W1_tr[j][i] * in[j]) >> 7;
            // using activate function
            int index = z1 << 12;
            if (z1 < -524288) index = 0x80000000;
            if (z1 > 262144) index = 0x7FFFFFFF;
            int index_sl = (index >> 22) & 1023;
            a1[i] = tabl_sigma[index_sl];
        }
        // second layer
        // Matrix-vector multiplication and getting over threshold operation
        for (i = 0; i < N; i++) {
            long long z2 = -b2_tr[i];
            for (j = 0; j < N; j++) z2 += ((long long) W2_tr[j][i] * a1[j]) >> 15;
            // using activate function
            int index1 = z2 << 12;
            if (z2 < -262144)
                index1 = 0x80000000;
            if (z2 > 262144) index1 = 0x7FFFFFFF;
            int index_sl1 = (index1 >> 22) & 1023;
            a2[i] = tabl_sigma[index_sl1];
        }
        // third layer
        // Matrix-vector multiplication and getting over threshold operation
        long long zout = -bout_tr[0];
        for (j = 0; j < N; j++) zout += ((long long) Wout_tr[j][0] * a2[j]) >> 15;
        // using activate function
        int index2 = zout << 12;
        if (zout < -262144) index2 = 0x80000000;
        if (zout > 262144) index2 = 0x7FFFFFFF;
        int index_sl2 = (index2 >> 22) & 1023;
        y_pred.P = tabl_sigma[index_sl2];
        long long zout1 = -bout_tr[1];
        for (j = 0; j < N; j++) zout1 += ((long long) Wout_tr[j][1] * a2[j]) >> 15;
        // using activate function
        int index3 = zout1 << 12;
        if (zout1 < -262144) index3 = 0x80000000;
        if (zout1 > 262144) index3 = 0x7FFFFFFF;
        int index_sl3 = (index3 >> 22) & 1023;
        y_pred.I = tabl_sigma[index_sl3];
        return y_pred;
}
