function VecFld = LGR(X, Y, conf,ind)
%%
%%Initialization
r = conf.r;
gama = conf.gama;
pz1 = conf.pz1;
lambda = conf.lambda;
a = conf.a;
MaxIter = conf.MaxIter;
ecr = conf.ecr;
theta = conf.theta;

%%
%%Input data
[N, D]=size(X); 
L = length(ind);
Xc = X(ind,:);
Yc = Y(ind,:);

%%
%tmp_X = unique(X, 'rows'); 
%idx = randperm(size(tmp_X,1)); 
%idx = idx(1:min(M,size(tmp_X,1)));ctrl_pts=tmp_X(idx,:);

%%
%%Preparation for f
%X2= sum(X.^2,2);   %N*1
%[~,distance, ~,neigbour]=compute_G_distance(X);
X2= sum(X.^2,2); %N*1
%[~,~,W ]=compute_G_distance(X2);
  distance = repmat(X2,1,N)+repmat(X2',N,1)-2*X*X';%Eu distance
  index = find(distance(:) < r);
  W = zeros(N*N,1);
  W(index) = exp(-distance(index)/r);
index = find(distance(:) < r);
W = zeros(N*N,1);
W(index) = exp(-distance(index)/r);

W = reshape(W,N,N);  
%[GG,nei_num,Nei] = Nei2G(W,Xc,gama);
Nei = (W>0) + 0;
nei_num = Nei2nei_num(Nei);
G  = Conpute_G(nei_num,Xc,Nei,gama);
S  = Nei2Sel(W);

M_matrix = S(1:size(G,1),:)'* G * S(1:size(G,1),:);
%J = eye(N);
%f =   zeros(L, D);
f =   zeros(N, D);
P = eye(L);%L*L
iter=1;  tecr=1; E=1;
sigma2=sum(sum((Yc-Xc).^2))/(L*D);
%%
%compute the graph laplacian matrix A 
%distance = repmat(X2,1,N)+repmat(X2',N,1)-2*X*X';
Dia = sum(W, 2);
A = Dia - W;
%D1 = diag(Dia.^(-0.5));
%A = eye(size(Dia))-D1*W*D1;
%%
% EM iteration
PS = [];
EE = [];
J = [eye(L),zeros(L,N-L)];
while ( (iter<MaxIter) && (abs(tecr) > ecr) && (sigma2 > 1e-8) )
 %% E-step.
    % Update P
    E_old = E;
    Tx = X + f;
    Tc = Tx(ind,:);
    [P1, E] = get_P(Yc, Tc, sigma2 ,pz1, a);  
    PS = [PS, P1];
    P = diag(P1);
    E1 = lambda/2*trace(f'*M_matrix*f);
    E2 = (1-lambda)/2*trace(f'*A*f);
    E = E + E1+E2;
    tecr=(E-E_old)/E;
    EE = [EE;E1,E2];
    
 %% M-step.
    % update C by solving linear system
    % update sigma^2 and gamma by tr(v_TPv)/D.tr(P)  and tr(P)/N
   f = (J'*P*J + lambda * sigma2*M_matrix + (1-lambda) * A )\(J'*P*(Yc-Xc));
    Vc = Yc - Xc - f(ind,:);
  sigma2 = trace(Vc'*P*Vc)/(D*trace(P));
    numcorr = length(find(P > theta));
    pz1=numcorr/N;
    if pz1 > 0.95, pz1 = 0.95; end
    if pz1 < 0.05, pz1 = 0.05; end
    iter=iter+1;
end
VecFld.X = X;
VecFld.Y = Y;
%VecFld.beta = beta;
VecFld.TX=  X +  f;
%VecFld.C=C;
%VecFld.ctrl_pts = ctrl_pts;
VecFld.P = diag(P);
VecFld.PS = PS;
VecFld.EE = EE;
VecFld.VFCIndex = find(VecFld.P > theta);





%{
for i = 1:N
    if i == 1
        f_i = S((1:nei_num(i)),:)*f;
        WB_temp =  (GG{i}'*GG{i})\GG{i}*f_i;
    else
        s = sum(nei_num(1:i-1));
        f_i = S((s+1:s+nei_num(i)),:)*f;
        f_i= [f_i;zeros(size(X,2),size(f_i,2))];
    end
         
            
     WB_temp =  (GG{i}'*GG{i})\GG{i}*f_i;
 end
%}
end

function G  = Conpute_G(nei_num,Xc,Nei,gama)
G = [];
[N,D] = size(Xc);
for i = 1:N
One_vec =ones(nei_num(i),1); 
Xi = Xc(Nei(i,:,:) == 1,:);
H=Xi'*Xi+gama*eye(D);
XHX=Xi/(H)*Xi';
c(i) =One_vec'*XHX*One_vec ;
A=eye(nei_num(i))-(XHX+...
(1/(nei_num(i)-c(i)))*(XHX*(One_vec*One_vec')*XHX-...
XHX*(One_vec*One_vec')-(One_vec*One_vec')*XHX+(One_vec*One_vec')));
G = blkdiag(G,A);
end
end
%{
function G  = Conpute_G(nei_num,Xc,Nei)
G = [];
for i = 1:N
One_vec =ones(nei_num(i),1); 
Xi = Xc(Nei(i,:,:) == 1,:);
H(i,:,:)=Xi'*Xi+gama*eye(D);
XHX(i,:,:)=( Xi*inv(reshape(H(i,:,:),size(H(i,:,:),2),size(H(i,:,:),3))) )*Xi';
c(i) =One_vec'*XHX(i,:,:)*One_vec ;
A(i,:,:)=eye(neigh(i))-(XHX(i,:,:)+...
(1/(neigh(i)-c(i)))*(XHX(i,:,:)*(One_vec*One_vec')*XHX(i,:,:)-...
XHX(i,:,:)*(One_vec*One_vec')-(One_vec*One_vec')*XHX(i,:,:)+(One_vec*One_vec')));
G = blkdiag(G,A(i,:,:));
end
end


function [G,nei_num,Nei] = Nei2G(W,X,gama)
Nei = (W>0) + 0;
[M,N] = size(Nei);
D= size(X,2);
G = cell(M);
nei_num = [];
for i = 1:M
    vec = Nei(i,:);
    ind = find(vec == 1);
    num = size(ind,2);
    G_temp = ones(num+D,D+1);
    
    G_temp(1:num,1:D)=(X(ind,:)-X(i,:));
    
    G_temp(num+1:num+D,1:D) = gama * eye(D);
    G_temp(num+1:num+D,D+1) = 0;
    
G{i} = G_temp; 
nei_num = [nei_num,num];
end

end 
%}
function [nei_num] = Nei2nei_num(Nei)

[M,N] = size(Nei);
nei_num = [];
for i = 1:M
    vec = Nei(i,:);
    ind = find(vec == 1);
    num = size(ind,2);
nei_num = [nei_num,num];
end

end 


function S  = Nei2Sel(W)
Nei = (W>0) + 0;
[M,N] = size(Nei);
S = [];
for i = 1:M
    vec = Nei(i,:);
    ind = find(vec == 1);
    S_temp = zeros(size(ind,2),N);
    for j = 1:size(ind,2)
        S_temp(j,ind(j)) = 1;
    end
S = [S;S_temp]; 
end
end

function TwoD_Tensor = Reshape(ThrD_Tensor)
[~,N2,N3] = size(ThrD_Tensor);
TwoD_Tensor = reshape(ThrD_Tensor,N2,N3);
end
function [P, E]=get_P(Y, Tx, sigma2 ,gamma, a)
% GET_P estimates the posterior probability and part of the energy.

D = size(Y, 2);
temp1 = exp(-sum((Y-Tx).^2,2)/(2*sigma2));
temp2 = (2*pi*sigma2)^(D/2)*(1-gamma)/(gamma*a);
P=temp1./(temp1+temp2);
E=P'*sum((Y-Tx).^2,2)/(2*sigma2)+sum(P)*log(sigma2)*D/2 - log(gamma)*sum(P) - log(1-gamma)*sum(1-P);
end