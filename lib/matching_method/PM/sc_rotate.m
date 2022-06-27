function [X,theta_radio] = sc_rotate(X,Y)
 mean_dist_global=[]; % use [] to estimate scale from the data
    nbins_theta=72;
    nbins_r=1;
    nsamp1=size(X,1);
    nsamp2=size(Y,1);
    ndum1=0;
    r_inner=0;
    r_outer=1000;
    out_vec_1=zeros(1,nsamp1);
    out_vec_2=zeros(1,nsamp2);
    Xk  = X;
    [BH1,mean_dist_1]=sc_computer(Xk',zeros(1,nsamp1),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_1);
    [BH2,mean_dist_2]=sc_computer(Y',zeros(1,nsamp2),mean_dist_1,nbins_theta,nbins_r,r_inner,r_outer,out_vec_2);

    % compute regularization parameter
    %beta_k=(mean_dist_1^2)*beta_init*r^(k-1);
    % compute pairwise cost between all shape contexts
    cost_value = [];
    for i=1:nbins_theta
        if i == 1
            temp = BH1;
            costmat=hist_cost_2(temp,BH2);
            cost_value = [cost_value mean(mean(costmat))];

        else
            temp = [BH1(:,((i-1)*nbins_r+1:nbins_theta*nbins_r)) BH1(:,1:((i-1)*nbins_r))];
            costmat=hist_cost_2(temp,BH2);
            cost_value = [cost_value mean(mean(costmat))];
        end      
    end
    
   [~,min_index] =min(cost_value);
 %  min_index = 18 - min_index;
   
   theta = 2*pi*(min_index-1)/nbins_theta;
   theta_radio = 360*min_index/nbins_theta;
   R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
   X = X*R';
   
   
%     if min_index ~=1
%         temp = [BH1(:,((min_index-1)*nbins_r+1:nbins_theta*nbins_r)) BH1(:,1:((min_index-1)*nbins_r))];
%     end
%     BH1 = temp;
%         