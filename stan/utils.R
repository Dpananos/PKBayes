library(glue)

compute_pk_profile<-function(Cl, ka, ke, times){
  y = 2.5*ke*ka/(Cl*(ke - ka)) * (exp(-ka*times) - exp(-ke*times))
  return(y)
}

profile_samples<-function(subject,times){
  
  which_params = c(glue('Cl[{subject}]'), glue('ka[{subject}]'), glue('ke[{subject}]'))

  params = theta_tilde[ , which_params]
  
  # rename the columns for convienience
  colnames(params) = c('Cl','ka','ke')
  
  # turn into a list of rows
  list_of_rows = array_branch(params, 1)
  
  # turn into a list of lists
  list_of_list = map(list_of_rows, array_branch)
  
  # give each list a named element t
  
  with_t = map(list_of_list,~{.x$times = times; .x})
  
  # Now compute profiles
  
  profiles = map(with_t, ~pmap_dbl(.x,compute_pk_profile))
  
  y = reduce(profiles, rbind)
  return(y)
  
}



f =no_condition %>% 
  bind_cols(predictions) %>% 
  filter(subjectids==12)

Y = profile_samples(subject=12,times=f$times)

f$Y = apply(Y,2,mean)


f %>% 
  ggplot()+
  geom_line(aes(times,Y), size = 2)+
  geom_line(aes(times, map_pred), color = 'red')


matplot(f$times, t(Y), type = 'l',col = 'black', lty=1, lwd = 0.1)

subject=12
which_params = c(glue('ka[{subject}]'), glue('ke[{subject}]'))

plot(theta_tilde[,which_params] )


f = fit %>% 
  spread_draws(ke[i], ka[i]) %>% 
  filter(i==12)


f %>% ungroup %>% select(ka,ke) %>% as.matrix() %>% plot
