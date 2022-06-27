% Copyright (c) Wei Lian (2010,2011), Department of computer science, Changzhi University, Changzhi, Shanxi province, China

function cost=SC_cost_fast_fun(edges,X,Y,Nang)

nbins_theta=12;
nbins_r=5;
r_inner=1/8;
r_outer=2;
%%%%%%%%%%%%%%%%%
Ny=size(Y,1);
Ne=size(edges,1);


dify=repmat(permute(Y,[1,3,2]),[1,Ny,1])-repmat(permute(Y,[3,1,2]),[Ny,1,1]); %every elements in y do the minus.edge vector
angy=atan2(dify(:,:,2),dify(:,:,1));%y edges angel

difx=X(edges(:,1),:)-X(edges(:,2),:);%every edges vector in x
angx=atan2(difx(:,2),difx(:,1)); %x edges angel


r_array=real(sqrt(dist2(X',X')));
mean_dist_1=mean(r_array(:));


[BHx1,mean_dist_2]=sc_compute(X(edges(:,1),:)',angx',mean_dist_1,nbins_theta,nbins_r,r_inner,r_outer,zeros(1,Ne));

[BHx2,mean_dist_2]=sc_compute(X(edges(:,2),:)',angx'+pi,mean_dist_1,nbins_theta,nbins_r,r_inner,r_outer,zeros(1,Ne));


cost1=zeros(Ne,Ny,Nang);
cost2=zeros(Ne,Ny,Nang);
for ii=1:Nang

    [BHy,mean_dist_2]=sc_compute(Y',(ii-1)*2*pi/Nang*ones(1,Ny),mean_dist_1,nbins_theta,nbins_r,r_inner,r_outer,zeros(1,Ny));

        cost1(:,:,ii)=hist_cost_2(BHx1,BHy);
        cost2(:,:,ii)=hist_cost_2(BHx2,BHy);

end


angy=rem(rem(angy,2*pi)+2*pi,2*pi);
angy=1+floor(angy*Nang/(2*pi));

cost_1=zeros(Ne,Ny,Ny);
cost_2=zeros(Ne,Ny,Ny);


for ii=1:Ne
    for jj=1:Ny
        cost_1(ii,jj,:)=cost1(ii,jj,angy(jj,:));
        cost_2(ii,jj,:)=cost2(ii,jj,angy(jj,:));        
    end
end
%%%%%

cost1=permute(cost_1,[2,3,1]);
cost2=permute(cost_2,[3,2,1]);

cost=zeros(Ny,Ny,Ne);

for ii=1:Ne-1
    if edges(ii,1)~=edges(ii+1,1)
        cost(:,:,ii)=cost1(:,:,ii);
    else
        cost(:,:,ii)=cost2(:,:,ii);
    end
end

if edges(Ne-1,1)~=edges(Ne,1)
    cost(:,:,Ne)=cost1(:,:,Ne);
else
    cost(:,:,Ne)=cost2(:,:,Ne);
end

