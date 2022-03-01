#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define N 12
#define M 25
#define T 9
void
nn_train(float x1[M], float x2[M], float x3[M], float x4[M], float P[M],
         float I[M], float W1[2][N], float Wout[N][2],
         float a1[N], float d_W1[2][N], float d_Wout[N][2], float b1[N],
         float d_b1[N], float bout[2], float d_bout[2],
         float z1[N], float zout[2], float y_pred[2], float delta_out[2],
         float delta1[N], float e1[N], float z2[N],
         float b2[N], float W2[N][N], float a2[N], float e2[N], float
         delta2[N], float d_b2[N], float d_W2[N][N],
         float **W1_tr, float **W2_tr, float **Wout_tr, float *b1_tr, float
         *b2_tr, float *bout_tr, float *y_dif,
         float *a2_dif, float *a1_dif, float x1_test[T], float x2_test[T],
         float x3_test[T], float x4_test[T],
         float P_test[T], float I_test[T]) {
    // входные и выходные значения
    /* нейросеть состоит из 4 слоёв:
    2 нейрона во входном слое;
    2 скрытых слоя с 12 нейронами в каждом;
    2 нейрона в выходном слое */
    int i, j, k, idx;
    // скорость обучения
    float lr;
    lr = 1;
    for ( i = 0; i < 4; i++ ) {
        // задание начальных значений порогов и весов для обучения
        // веса между входным и 1 скрытым слоем
        for ( j = 0; j < N; j++ ) W1[i][j] = (float) (rand() % 1000) / 1000;
    }
    // пороги выходного слоя
    for ( i = 0; i < 2; i++ ) bout[i] = 4;
    for ( i = 0; i < N; i++ ) {
        // пороги второго скрытого слоя
        b2[i] = 4;
        // веса второго скрытого слоя
        for ( j = 0; j < N; j++ ) W2[i][j] = (float) (rand() % 1000) / 1000;
    }
    for ( i = 0; i < N; i++ ) {
        // пороги первого скрытого слоя
        b1[i] = (float) (rand() % 1000) / 1000;
        // веса выходного слоя
        for ( j = 0; j < 2; j++ ) Wout[i][j] = (float) (rand() % 1000) / 1000;
    }
    for ( k = 0; k < 5000; k++ ) {
        for ( idx = 0; idx < M; idx++ ) {
            /* просвоение массиву входных сигналов и целевому массиву
           соответствующих выходных
            сигналов из обучающей выборки */
            float in[4] = {x1[idx], x2[idx], x3[idx], x4[idx]};
            float y_true[2] = {P[idx], I[idx]};
            // прямое распространение
            // первый слой
            // Операция матрично-векторного умножения и учета порога
            for ( i = 0; i < N; i++ ) {
                z1[i] = -b1[i];
                for ( j = 0; j < 4; j++ ) z1[i] += W1[j][i] * in[j];
            }
            // применение активационной функции
            for ( i = 0; i < N; i++ ) {
                a1[i] = 1 / (1 + expf(-z1[i]));
            }
            // второй слой
            // операция матрично-векторного умножения и учёта порога
            for ( i = 0; i < N; i++ ) {
                z2[i] = -b2[i];
                for ( j = 0; j < N; j++ ) z2[i] += W2[j][i] * a1[j];
            }
            // применение активационной функции
            for ( i = 0; i < N; i++ ) {
                a2[i] = 1 / (1 + expf(-z2[i]));
            }
            // выходной слой
            // Операция матрично-векторного умножения и учета порога
            for ( i = 0; i < 2; i++ ) {
                zout[i] = -bout[i];
                for ( j = 0; j < N; j++ ) zout[i] += Wout[j][i] * a2[j];
            }
            // применение активационной функции
            for ( i = 0; i < 2; i++ ) {
                y_pred[i] = 1 / (1 + expf(-zout[i]));
            }
            // вычисление величин коррекции выходного слоя
            for ( i = 0; i < 2; i++ ) {
                y_dif[i] = y_pred[i] * (1 - y_pred[i]);
                delta_out[i] = (y_pred[i] - y_true[i]) * y_dif[i];
                d_bout[i] = -delta_out[i];
                for ( j = 0; j < N; j++ ) d_Wout[j][i] = delta_out[i] *
                                                         a2[j];
            }
            //вычисление величин коррекции между 2 и 3 слоем
            for ( i = 0; i < N; i++ ) {
                e2[i] = 0;
                for ( j = 0; j < 2; j++ ) e2[i] += Wout[i][j] * delta_out[j];
                a2_dif[i] = a2[i] * (1 - a2[i]);
                delta2[i] = e2[i] * a2_dif[i];
                d_b2[i] = -delta2[i];
                for ( j = 0; j < N; j++ ) d_W2[j][i] = delta2[i] * a1[j];
            }
            // вычисление величин коррекции входного слоя
            for ( i = 0; i < N; i++ ) {
                e1[i] = 0;
                for ( j = 0; j < N; j++ ) e1[i] += W2[i][j] * delta2[j];
                a1_dif[i] = a1[i] * (1 - a1[i]);
                delta1[i] = e1[i] * a1_dif[i];
                d_b1[i] = -delta1[i];
                for ( j = 0; j < 4; j++ ) d_W1[j][i] = delta1[i] * in[j];
            }
            // обновление параметров
            for ( i = 0; i < 4; i++ ) {
                for ( j = 0; j < N; j++ ) W1[i][j] = W1[i][j] - lr *
                                                                d_W1[i][j];
            }
            for ( i = 0; i < 2; i++ ) bout[i] = bout[i] - lr * d_bout[i];
            for ( i = 0; i < 2; i++ ) {
                for ( j = 0; j < N; j++ ) Wout[j][i] = Wout[j][i] - lr *
                                                                    d_Wout[j][i];
            }
            for ( i = 0; i < N; i++ ) {
                b1[i] = b1[i] - lr * d_b1[i];
                b2[i] = b2[i] - lr * d_b2[i];
            }
            for ( i = 0; i < N; i++ ) {
                for ( j = 0; j < N; j++ ) W2[i][j] = W2[i][j] - lr *
                                                                d_W2[i][j];
            }
        }
    }
    float err = 0;
    // тестирование
    for ( idx = 0; idx < T; idx++ ) {
        // Формарование массивов входных сигналов и целевых меток нейросети
        float in[4] = {x1[idx], x2[idx], x3[idx],
                       x4_test[idx]};
        float y_true[2] = {P[idx], I[idx]};
        // Операция матрично-векторного умножения и учета порога
        for ( i = 0; i < N; i++ ) {
            z1[i] = -b1[i];
            for ( j = 0; j < 4; j++ ) z1[i] += W1[j][i] * in[j];
        }
        // применение активационной функции
        for ( i = 0; i < N; i++ ) a1[i] = 1 / (1 + expf(-z1[i]));
        // второй слой
        // операция матрично-векторного умножения и учёта порога
        for ( i = 0; i < N; i++ ) {
            z2[i] = -b2[i];
            for ( j = 0; j < N; j++ ) z2[i] += W2[j][i] * a1[j];
        }
        // применение активационной функции
        for ( i = 0; i < N; i++ ) a2[i] = 1 / (1 + expf(-z2[i]));
        // выходной слой
        // Операция матрично-векторного умножения и учета порога
        for ( i = 0; i < 2; i++ ) {
            zout[i] = -bout[i];
            for ( j = 0; j < N; j++ ) zout[i] += Wout[j][i] * a2[j];
        }
        // применение активационной функции
        for ( i = 0; i < 2; i++ ) {
            char *str;
            if (i == 0) {str = "P";}
            else {str = "K";}
            y_pred[i] = 1 / (1 + expf(-zout[i]));
            printf("%c_out = %.2f, %c_true = %.2f\n", *str, y_pred[i], *str,
                   y_true[i]);
            // вычисление суммы для среднего квадратичного отклонения
            err += pow(y_pred[i] - y_true[i], 2);
        }
        printf("\n");
    }
    err = err / (T * 2);
    printf("error of learning: %.3f%%", err * 100);
    // вывод массивов весов и порогов из функции
    for ( i = 0; i < 2; i++ ) {
        bout_tr[i] = bout[i];
    }
    for ( i = 0; i < 4; i++ ) {
        for ( j = 0; j < N; j++ ) W1_tr[i][j] = W1[i][j];
    }
    for ( i = 0; i < N; i++ ) {
        b1_tr[i] = b1[i];
        b2_tr[i] = b2[i];
        for ( j = 0; j < 2; j++ ) Wout_tr[i][j] = Wout[i][j];
        for ( j = 0; j < N; j++ ) W2_tr[i][j] = W2[i][j];
    }
}