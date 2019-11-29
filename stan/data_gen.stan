// Stan code to generate data for simulation study

data{
    int n_subjects; // Number of subjects
    int n_continuous_predictors;
    int n_binary_predictors;
    real<lower=0, upper=1> binary_prob;
    vector[3] beta_baseline; //baseline values for the PK parameters on the log scale.
    vector[3] beta_cov; //covariance for the betas.  Assume indpendent.  Will be diagonal.
    vector[3] rfx_cov;
    int use_normal_rfx;

}
transformed data{
    int p=1+n_continuous_predictors+n_binary_predictors;
}
generated quantities{
    //Generating Covariates
    vector[n_continuous_predictors] mu_x=rep_vector(0, n_continuous_predictors);
    matrix[n_continuous_predictors, n_continuous_predictors] Sigma_x = lkj_corr_rng(n_continuous_predictors,1.0);
    matrix[n_subjects, n_continuous_predictors] X_cont = rep_matrix(0, n_subjects, n_continuous_predictors);
    matrix[n_subjects, n_binary_predictors] X_bin = rep_matrix(0, n_subjects, n_binary_predictors);
    matrix[n_subjects, p] X = rep_matrix(0, n_subjects, p);
    vector[n_subjects] col_of_ones = rep_vector(1, n_subjects);

    //Generating regression coefficients
    matrix[p, 3] beta = rep_matrix(0, p, 3); //3 PK coefficients

    //Generating pk params
    matrix[n_subjects, 3] pk = rep_matrix(0, n_subjects, 3);
    matrix[n_subjects,3] z_pk = rep_matrix(0, n_subjects, 3);
    matrix[3,3] rfx_sigma = rep_matrix(0, 3, 3);


    //First, we generate subject predictors
    for(i in 1:n_subjects){
        X_cont[i, 1:n_continuous_predictors] = multi_normal_rng(mu_x, Sigma_x)';
        X_bin[i] = to_row_vector(bernoulli_rng(rep_vector(binary_prob, n_binary_predictors))); 
    }
    X = append_col(append_col(col_of_ones, X_cont), X_bin);

    //Next, we generate coefficients for the regression
    beta[1] = beta_baseline';
    for (i in 2:p){
        beta[i] = multi_normal_rng(rep_vector(0, 3), diag_matrix(beta_cov))';
    }

    //Next, we generate the random effects for the pk params
    for(i in 1:n_subjects){
        if (use_normal_rfx==1){
            z_pk[i] = multi_normal_rng(rep_vector(0,3), diag_matrix(rep_vector(1,3)))';
        }
        else{
            z_pk[i] = to_row_vector(student_t_rng(5, rep_vector(0, 3), 1));
        }
        
    }
    pk = exp(X*beta + z_pk*rfx_sigma);
}