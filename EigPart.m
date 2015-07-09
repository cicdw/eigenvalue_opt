function [labs,objs,VS,purities,tf,avgNumIters] = EigPart(L,labs,c,gtlabs,semiLabs)
% partitions the vertices of a graph with Laplacian L
% labs are initial labels, c is a parameter,
% semiLabs are labels for semi-supervised learning
% gtlabs are (optional) ground truth labels
%
% labs = randi(k,n,ni);
% l = eigs(L,[],k+1,'sm'); c = 5*l(2);
%

tic
if nargin< 5, semiSup = 0; else semiSup = 1; end
if nargin<4, gtCompare = 0; purities=0; else gtCompare=1; end

maxNumIters = 100;
objTol = 1e-10;
eigTol = 1e-4;

n = size(L,1);
ni = size(labs,2);
k = max(labs(:));
objs = inf(ni,1); VS=zeros(n,k); purities = objs; numIters = objs;

for ll = 1:ni, % number of initializations
    if semiSup, labs(semiLabs(:,1),ll) = semiLabs(:,2); end
        
    for ii = 1:maxNumIters,
        lams = zeros(k,1);
        for jj = 1:k,
            if ii==1,
                opts = struct('issym',1,'tol',eigTol);
                [v,lam] = eigs(L + c*sparse(1:n,1:n,labs(:,ll)~=jj,n,n),[],1,'sm',opts);
                % sm is faster for small datasets 
                % sa is faster for MNIST, large datasets
                
                % for r=1 only. 
                %[v,lam] = eigs(L-speye(n,n) + c*sparse(1:n,1:n,labs(:,ll)~=jj,n,n),[],1,'lm',opts);
                
                %RC_Op = RayleighChebyshev();
                %RC_Op.clearVerbose();
                %[v,lam]  = RC_Op.getMinEigenSystem(L + c*sparse(1:n,1:n,labs(:,ll)~=jj,n,n),1,1e-4,struct('bufferSize',2));
            else
                opts = struct('v0',VS(:,jj),'issym',1,'tol',eigTol);
                [v,lam] = eigs(L + c*sparse(1:n,1:n,labs(:,ll)~=jj,n,n),[],1,'sm',opts);

                %RC_Op = RayleighChebyshev();
                %RC_Op.clearVerbose();
                %[v,lam]  = RC_Op.getMinEigenSystem(L + c*sparse(1:n,1:n,labs(:,ll)~=jj,n,n),1,1e-4,struct('bufferSize',2,'initialSubspace',VS(:,jj)));
            end
            lams(jj) = lam; VS(:,jj) = v;
        end
        obj = sum(lams); if ii > 1 && norm(objs(ll) - obj)<objTol, break; end
        objs(ll) = obj; [~,lab] = max(abs(VS),[],2); labs(:,ll) = lab;
        
        if semiSup, labs(semiLabs(:,1),ll) = semiLabs(:,2); end
        
        if length(unique(labs(:,ll)))<k, disp('label lost'); break; end
        
        if gtCompare,
            [purity, ~, ~, ~] = ComputeClusterPurity(labs(:,ll),gtlabs,k,k);
            fprintf('ic %i, iter %i, J = %0.5g, purity = %0.5g, time=%0.4g \n',ll,ii,obj,purity,toc);
            purities(ll) = purity;
        else
            fprintf('ic %i, iter %i, J = %0.5g, time=%0.4g \n',ll,ii,obj,toc);
        end
    end
    numIters(ll) = ii-1;
    fprintf('\n')
    %if purity>.9, break; end;
end

tf = toc;
avgNumIters = mean(numIters);
