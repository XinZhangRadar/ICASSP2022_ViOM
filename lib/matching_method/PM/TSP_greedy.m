% Copyright (c) Wei Lian (2010,2011), Department of computer science, Changzhi University, Changzhi, Shanxi province, China

function X=TSP_greedy(X)




N=size(X,1);

dif=repmat(permute(X,[1,3,2]),[1,N,1])-repmat(permute(X,[3,1,2]),[N,1,1]);
dist=sqrt(sum(dif.^2,3));


for ii=1:N
    for jj=ii:N
        dist(ii,jj)=inf;
    end
end

edge=[];

while 1
    
    min_dist=min(dist(:));
    [indx,indy]=find(dist==min_dist);
    indx=indx(1);
    indy=indy(1);
    
    dist(indx,indy)=inf;
    
    xinedge=find(edge==indx);
    if size(xinedge)<2
        yinedge=find(edge==indy);
        if size(yinedge)<2
            if size(edge,1)==N-1
                edge=[edge;[indx,indy]];
                break;
            else
                
                edge2=edge;
                indt=indx;
                while 1
                    [ind1,ind2]=find(edge2==indt);
                    if ind1
                        indt=edge2(ind1,3-ind2);
                        edge2(ind1,:)=[inf,inf];
                    else
                        break;
                    end
                end
                if indt~=indy
                    edge=[edge;[indx, indy]];
                end
            end
        end
    end
end


Z=zeros(N,2);

ind1=1;
ind2=1;
ind=edge(1,1);
for ii=1:size(edge,1)     
    Z(ii,:)=X(ind,:);
    ind=edge(ind1,3-ind2);    
    edge(ind1,:)=[inf,inf];
    [ind1,ind2]=find(edge==ind);  
end


X=Z;



