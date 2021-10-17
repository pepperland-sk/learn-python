data {
   /* ... declarations ... */
   int n_sample;     // サンプルサイズ
   real y[n_sample]; // 観測値
}

parameters {
   /* ... declarations ... */
   real mu_zero;      // 状態の初期値
   real mu[n_sample]; // 状態の推定値
   real<lower=0> s_w; // 過程誤差の分散
   real<lower=0> s_v; // 観測誤差の分散

}

model {
   /* ... declarations ... statements ... */
   mu[1] ~ normal(mu_zero, sqrt(s_w));

   mu[2:n_sample] ~ normal(mu[1:(n_sample - 1)], sqrt(s_w));

   y ~ normal(mu, sqrt(s_v));
//    for(i in 2:n_sample) {
//        mu[i] ~ normal(mu[i-1], sqrt(s_w));
//    }

//    for(i in 1:n_sample) {
//        mu[i] ~ normal(mu[i], sqrt(s_v));
//    }
}