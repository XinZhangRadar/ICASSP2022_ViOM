function index_result = demoMR()
%%This is a demo for removing outliers. In this demo, the SIFT matches  is known in advance.
%clear;
%close all;
%warning off all;

normalize = 1;
conf.lambda1 = 3;
conf.lambda2 = 0.05;
conf.r = 0.5;
conf.a = 4;

%conf.beta = 0.1

show_result = 0;

%load save_chinese_occlusion_6_97.mat;
addpath('/home/zhangxin/PM/sc/');

%Y = [351,230; 331,293; 219,328;323,351;416,431; 200,388; 309,410; 361,173;];
%X = [580,348; 563,407;477,329; 533,544;578,687; 669,433; 670,579; 503,676; 620,175;609,233;460,387;585,293; ];
%46-5
%X = [420,159;172,261;451,55;82,347;52,453;118,384;487,92];
%Y = [354,448;200,231;157,189;397,490;211,185;267,246;408,446];
%46-23
% X = [420,159;172,261;451,55;82,347;52,453;118,384;487,92];
% Y = [95,117;105,82;78,50;126,171;151,418;179,449;169,486];
% 
%46-6
 %X = [420,159;172,261;451,55;82,347;52,453;118,384;487,92];
 %Y = [346,375;113,397;218,312;394,360;160,381;156,330;391,309];

load XY;
%X = y1;
%Y = y2a;
tic;
conf.M = ceil(size(X,1)/2);
conf = LLT_init(conf);
%figure
%plot(X(:,1),X(:,2),'b+',Y(:,1),Y(:,2),'ro');
%title( 'original data');
X_save = X;
Y_save = Y;





 [X2,theta_radio] = sc_rotate(X,Y);
 %fprintf('旋转%f度',theta_radio);
% 
% figure
% plot(X2(:,1),X2(:,2),'b+',Y(:,1),Y(:,2),'ro');
% title( 'original data');
%X2 = X;
   
index_result = 1:size(Y,1);    

[X2,Y2,V,ind,ifxmore,out] = sc(X2,Y,X2);

index_result = index_result(out);
if ifxmore
    X2 = X2(ind,:);
end
lenX = size(X,1);
lenY = size(Y,1);
if lenX < lenY
    Xtmp=NaN*ones(lenY,2);
    Xtmp(1:lenX,:)=X;
    X=Xtmp(out,:);
else
    X=X(out,:);
end
%X = X(out,:);
normal.xm=0; normal.ym=0;
normal.xscale=1; normal.yscale=1;

if normalize
    [nX, nY, normal]=norm_ind(X2,Y2,ind);
end


if ~exist('conf'), conf = []; end



VecFld=MR(nX, nY, conf,ind);
%VecFld=LGR(nX, nY, conf,ind);





Xtemp = [];
means = [];
for iii = 1:30
    if normalize,VecFld.TX=(VecFld.TX)*normal.yscale+repmat(normal.ym,size(VecFld.TX,1),1);end
    Xk = VecFld.TX;
   
  [Xk,theta_radio] = sc_rotate(Xk,Y2);
  fprintf('旋转%f度',theta_radio);
    
   [X2,Y2,V,ind,~,out] = sc(Xk,Y2,V);
   index_result = index_result(out);
   %X2 = Xk;
    if ifxmore
        X2 = X2(ind,:);
    end
    %X = X(out,:);
    YY = Xk(ind,:);
    mea = sum(sqrt(sum((YY-V).^2,2)))/size(YY,1);
    means = [means;mea];
    Xtemp = [Xtemp;Xk];

    if normalize
        [nX, nY, normal]=norm_ind(X2,Y2,ind);
    
    else
    nX = X2;
    nY = Y2;
end



    VecFld=MR(nX, nY, conf,ind);
    %VecFld=LGR(nX, nY, conf,ind);



end

if normalize,VecFld.TX=(VecFld.TX)*normal.yscale+repmat(normal.ym,size(VecFld.TX,1),1);end
Xk = VecFld.TX;
toc

if show_result
   
    figure
    plot(Xk(:,1),Xk(:,2),'b+',Y(:,1),Y(:,2),'ro');
    title( 'registration result');
end

 
 % figure;hold on;
 %plot(Y(:,1),Y(:,2),'b+','LineWidth',2);
 %plot(X(:,1),X(:,2),'r*','LineWidth',2);
 %plot([X_save(index_result(1:size(Y,1)),1) Y(:,1)]',[X_save(index_result(1:size(Y,1)),2),Y(:,2)]','k-','LineWidth',2);
end