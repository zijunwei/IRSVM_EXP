classdef M_IRSVM
    % Instance-Ranking SVM for MIL
    % By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
    % Created: 15-Aug-2013
    % Last modified: 11-Oct-2013

    methods (Static)
        % Train IR-SVM
        % Bs: 1*n cell structure for bags of instances. Bs{i} is a d*m_i matrix for m_i instances
        % lb: n*1 label vector, lb(i) is 1 or -1.
        % C: C for SVM
        % opts:
        %   opts.initOpt: default 'mean';
        %   opts.nIter1:  default 10
        %   opts.isMISVM: 1, this is MISVM, 0: this is IRSVM. Default is 0.
        % This solves the optimization problem
        %   min_{w,s,b} 0.5*(w'*w) + C*sum_i xi_i
        %          s.t. max_P lb(i)*(w'*Bs{i}*P*s + b) >= 1 - xi_i
        %               xi_i >= 0
        %               s(1) >= ... >= s(m) >= 0
        %               sum_i s_i == 1 (equivalent to constraint sum_i s_i <= 1)        
        function [w, b, s, objVal, iterInfo] = train(Bs, lb, C, opts)
            m = min(100, max(cellfun(@(x)size(x,2), Bs))); % sample m instances per bag                                     
            
            n = length(Bs);
            d = size(Bs{1},1);
            
            % reorder the bags, putting positive first
            posIdxs = lb == 1;
            nPos = sum(posIdxs);
            nNeg = n - nPos;
            Bs = [Bs(posIdxs), Bs(~posIdxs)]; 
            lb = [ones(nPos,1); -ones(nNeg,1)]; 
            if length(C) > 1
                C = [C(posIdxs); C(~posIdxs)];
            end;
                        
            avgVals = zeros(1, nPos);
            for i=1:nPos
                ml_progressBar(i, nPos);
                A  = Bs{i}'*Bs{i};
                avgVals(i) = (sum(A(:)) - sum(diag(A)))/(size(A,1)*(size(A,1)-1)*mean(diag(A)));
            end;
            avgCosineVal = mean(avgVals);
            fprintf('avgCosineVal: %g\n', avgCosineVal);

            
            % structure to make sure make each bag has exactly m instances
            % we replicate the indexes instead of the data 
            irIdxs = cell(1, n); 
            
            try 
                initOpt = opts.initOpt;
            catch %#ok<CTCH>
                initOpt = 'mean';
            end
            
            try
                nIter1 = opts.nIter1;            
            catch %#ok<CTCH>
                nIter1 = 10;
            end;
            
            try 
                isMISVM = opts.isMISVM;
            catch %#ok<CTCH>
                isMISVM = 0;
            end;
            
            if strcmpi(initOpt, 'wb')
                w = opts.w;
                b = opts.b;      
                s = zeros(m,1); s(1) = 1;
            else
                % initialization             
                BR = zeros(d, n); % bag representative
                if strcmpi(initOpt, 'mean')                
                    for i=1:n
                        BR(:,i) = mean(Bs{i}, 2); % i.e., s = 1/m*ones(m,1)
                    end;            
                    s = 1/m*ones(m,1);
                elseif strcmpi(initOpt, 'first')                
                    for i=1:n
                        BR(:,i) = Bs{i}(:,1);
                    end;                        
                    s = zeros(m,1); s(1) = 1;
                else
                    error('unknow init option');
                end;
                %[w, b] = M_IRSVM.linearSvm_primal(BR, lb, 1:n, C);                      
                [w, b] = M_IRSVM.linearSvm(BR, lb, {}, C);
            end
                        
            
            % coordinate descent for permutation of positive bags
            iter1objs = zeros(1, nIter1);
            [iterInfo.ss, iterInfo.iter2objs] = deal(cell(1, nIter1));
            
            for iter1=1:nIter1                
                fprintf('=======> %s, iter: %d\n', opts.methodName, iter1);
                % update the order of instances
                IS = zeros(m, n); % instance score
                for i=1:n                    
                    score_i = Bs{i}'*w;
                    [irIdxs{i}, IS(:,i)] = M_IRSVM.scoreSample(score_i, m);                    
                end;                
                
                % update s, by solving a quad prog
                objVal = M_IRSVM.cmpObj(Bs, lb, C, w, b, s);
                
                fprintf('  Before updating s: %.6f\n', objVal);        
                if exist('isMISVM', 'var') && isMISVM
                    s = zeros(m, 1); s(1) = 1;
                    objVal = M_IRSVM.cmpObj(Bs, lb, C, w, b, s);
                else                                        
                    [s, ~, objVal] = M_IRSVM.update_s(IS, lb, C, avgCosineVal);
                    objVal = objVal + 0.5*(w'*w);  
                    fprintf('  s: '); fprintf('%g ', s(1:15)); fprintf('\n');
                end;
                              
                iter1objs(iter1) = objVal;
                iterInfo.ss{iter1} = s;                
                iterInfo.iter1objs(iter1) = objVal;
                fprintf('  After updating s:  %.6f\n', objVal); 
                
                fprintf('  Updaing wb using constraint generation:\n');
                %[w, b, objVal] = M_IRSVM.update_wb_old(Bs, nPos, s, irIdxs, C, iter1);
                opts.dispMsg = sprintf('%s iter1: %d,', opts.methodName, iter1);
                [w, b, objVal, iter2objs] = M_IRSVM.update_wb(Bs, nPos, s, irIdxs, C, opts);
                iterInfo.iter2objs{iter1} = iter2objs;
                
                if (iter1objs(iter1) - objVal) < 1e-3*iter1objs(iter1) 
                    fprintf('  Decrease in obj val is small, terminating outter loop\n');
                    break;
                end;
                
            end;            
        end;
        
        % Bs: 1*n cell for n bags, the first nPos bags are positive
        % irIdxs: 1*n cell for instance selection (replication) indicators
        % s: the current mixing coefficients
        % opts: options
        %   nThread, set opts.nThread=1 for single thread. Don't set opts.nThread for maximum # of threads. 
        %   DisplayFunc: call-back display function. Set to [] for no display. Don't set it for
        %   default display
        %   dispMsg: display message apended to the begining of iteration printout
        % This code maintain a single cplex object and iteratively add/remove constraints
        % This contrasts with update_wb_old which uses a series of cplex objects for QP.
        % This code is faster than upbdate_ws_old;
        function [w, b, objVal, iterObjs] = update_wb(Bs, nPos, s, irIdxs, C, opts)
            tol = 1e-6;
            
            d = size(Bs{1},1);
            n = length(Bs);
            nNeg = n - nPos;
            m = length(s);
            
            % update bag representative for positive
            posBR = zeros(d, nPos);
            for i=1:nPos
                posBR(:,i) = Bs{i}(:, irIdxs{i})*s;
            end;
            
            % reset the representative for negative
            negBR = zeros(d, nNeg);
            in2bagMap = 1:nNeg; % mapping from neg data instances to neg bags
            for i=1:nNeg
                negBR(:,i) = Bs{i+nPos}(:, irIdxs{i+nPos})*s;
            end;
            
            % Set up the QP
            % variable x = [w; b; xi] = [w_1; ...; w_d; b; xi_1; ...; xi_k];
            cplex = Cplex('pSVM.update_wb');
            cplex.Model.sense = 'minimize';
            
            d1  = d+1;
            d1n = d1+n; % number of variables
            % Add linear part of the objective
            % Add to the objective: sum_i C_i*xi_i
            obj = zeros(d1n,1);
            obj(d1+1:d1n) = C;
            % cplex.Model.obj = obj; % this syntax doesn't work, use the below
            cplex.addCols(obj);            
            
            % Add to objective: 0.5*||w||^2;
            Q = zeros(d1n, d1n); % allocate d non-zero entry for Q
            for j=1:d
                Q(j,j) = 1;
            end
            cplex.Model.Q = sparse(Q);
            
            % Add constraint: xi_i >= 0
            lbound = -inf(d1n,1);
            lbound(d1+1:d1n) = 0;
            cplex.Model.lb = lbound;
            
            % Add constraints for positive examples: w'*posBR(:,i) + b >= 1 -xi_i 
            constrVecs = sparse([posBR', ones(nPos,1), eye(nPos), zeros(nPos, nNeg)]);
            cplex.addRows(ones(nPos,1), constrVecs, inf(nPos,1));
            
            
            % Callback function for display
            try
                cplex.DisplayFunc = opts.dispFunc; %@disp;
            catch %#ok<CTCH>
            end;
            
            % solve
            try
                cplex.Param.threads.Cur = opts.nThread; % use a single thread
            catch %#ok<CTCH>
            end;
            
            try
                dispMsg = opts.dispMsg;
            catch %#ok<CTCH>
                dispMsg = '';
            end;
            
            try 
                nIter2 = opts.nIter2;
            catch %#ok<CTCH>
                nIter2 = 30;
            end;
            
            % Update w, using constraint generation, for permuations of instances in negative bags
            for iter2=1:nIter2
                fprintf('  %s iter: %d\n', dispMsg, iter2);
                                
                % Add constraints for new negative examples
                % w'*negBR(:,i) + b <= -1 + xi_{in2bagMap(i) + nPos}
                constrVecs = negBR';
                nNeg2 = size(negBR,2); % number of negative instances (not bags)              
                SM = zeros(nNeg2, nNeg);
                linIdx = sub2ind([nNeg2, nNeg], 1:nNeg2, in2bagMap);
                SM(linIdx) = -1;
                constrVecs = sparse([constrVecs, ones(nNeg2,1), zeros(nNeg2, nPos), SM]);
                cplex.addRows(-inf(nNeg2, 1), constrVecs, - ones(nNeg2,1));                
                fprintf('    nNewConstr: %d, total: %d\n', nNeg2, size(cplex.Model.A, 1));
                
                cplex.solve();
                x = cplex.Solution.x;
                objVal = cplex.Solution.objval;
                w = x(1:d);
                b = x(d1);
                xi = x(d1+1:end); % slack
                negXi = xi(nPos+1:end);
                
                fprintf('    objVal: %g\n', objVal);
                iterObjs(iter2) = objVal;
                
                dual = cplex.Solution.dual;                
                negDual = dual(nPos+1:end); % dual for negative constraint
                untightNegIdxs = find(abs(negDual) < tol); % indexes of untight constraints
                if ~isempty(untightNegIdxs)
                    untightNegIdxs = nPos + untightNegIdxs;                    
                    cplex.delRows(untightNegIdxs);
                end
                fprintf('    nRemoveConstr: %d\n', length(untightNegIdxs));
                                
                % generate most violated constraints for negative
                newNegBR = zeros(d, nNeg);
                newNegScore = zeros(nNeg, 1);
                for i=nPos+1:n
                    score_i = Bs{i}'*w;
                    [irIdxs_i, score_i2] = M_IRSVM.scoreSample(score_i, m);
                    newNegBR(:,i-nPos) = Bs{i}(:, irIdxs_i)*s;
                    newNegScore(i-nPos) = score_i2'*s;
                end;                
                newNegScore = newNegScore + b;
                newIn2bagMap = 1:nNeg;
                
                newConstrIdxs = newNegScore > -1 + tol + negXi;
                nNewConstr = sum(newConstrIdxs);
                if nNewConstr == 0
                    fprintf('    No new constraint, termininating CG\n');
                    break;
                end
                
                in2bagMap = newIn2bagMap(newConstrIdxs); 
                negBR = newNegBR(:,newConstrIdxs);                
            end
        end
        
        
        
        % Bs: 1*n cell for n bags, the first nPos bags are positive
        % irIdxs: 1*n cell for instance selection (replication) indicators
        % s: the current mixing coefficients
        % This performs constraint generation with a series of QP. Each QP initializes a different
        % cplex problem. This code is replaced by update_wb, which is faster.
        function [w, b, objVal] = update_wb_old(Bs, nPos, s, irIdxs, C, iter1)
            nIter2 = 20;
            tol = 1e-4;
            
            d = size(Bs{1},1);
            n = length(Bs);
            nNeg = n - nPos;
            m = length(s);
            
            % update bag representative for positive
            posBR = zeros(d, nPos);
            for i=1:nPos
                posBR(:,i) = Bs{i}(:, irIdxs{i})*s;
            end;
            
            % reset the representative for negative
            negBR = zeros(d, nNeg);
            in2bagMap = 1:nNeg; % mapping from neg data instances to neg bags
            for i=1:nNeg
                negBR(:,i) = Bs{i+nPos}(:, irIdxs{i+nPos})*s;
            end;
            
            % Update w, using constraint generation, for permuations of instances in negative bags
            for iter2=1:nIter2
                % figure out the slack group from the in2bagMap
                slackGrp = {};
                grpCnt = 0;
                for j=1:nNeg
                    idxs = find(in2bagMap == j); % instances from bag j
                    if length(idxs) > 1
                        grpCnt = grpCnt + 1;
                        slackGrp{grpCnt} = idxs + nPos; %#ok<*AGROW>
                    end;
                end;
                
                if isscalar(C)
                    C4primal = C;
                    C4dual = C;
                else
                    C4primal = C(1:nPos+max(in2bagMap));  % one for each slack variable (group)
                    C4dual = C([1:nPos, nPos+in2bagMap]); % one for each data point
                end
                
                % optimize the primal
                [w, b, objVal] = M_IRSVM.linearSvm_primal([posBR, negBR], ...
                    [ones(nPos,1); -ones(size(negBR,2),1)], [1:nPos, nPos + in2bagMap], C4primal);
                
                % optimize the dual
                %  [w, b, ~, objVal] = M_IRSVM.linearSvm([posBR, negBR], ...
                %  [ones(nPos,1); -ones(size(negBR,2),1)], slackGrp, C4dual);
                
                
                fprintf('  iter: %d-%d, objVal: %g\n', iter1, iter2, objVal);
                
                score4neg = negBR'*w + b;
                
                % find the max score for negative bags
                score4negBag = -inf(nNeg,1);
                for j=1:nNeg
                    idxs = find(in2bagMap == j); % instances from bag j
                    if ~isempty(idxs)
                        score4negBag(j) = max(score4neg(idxs));
                    end;
                end
                
                
                % remove untight negative constraints
                removeIdxs = score4neg < -1 - tol;
                negBR(:, removeIdxs) = [];
                in2bagMap(removeIdxs) = [];
                fprintf('  nNegRemoved: %d\n', sum(removeIdxs));
                
                % generate most violated constraints for negative
                newNegBR = zeros(d, nNeg);
                for i=nPos+1:n
                    score_i = Bs{i}'*w;
                    irIdxs{i} = M_IRSVM.scoreSample(score_i, m);
                    newNegBR(:,i-nPos) = Bs{i}(:, irIdxs{i})*s;
                end;
                
                score4newNeg = newNegBR'*w + b;
                
                newConstrIdxs = and(score4newNeg > score4negBag + tol, score4newNeg > -1 + tol);
                
                nNewConstr = sum(newConstrIdxs);
                if nNewConstr == 0
                    fprintf('    No new constraint, termininating CG\n');
                    break;
                end
                
                newIn2bagMap = 1:nNeg;
                in2bagMap = [in2bagMap, newIn2bagMap(newConstrIdxs)]; %#ok<AGROW>
                negBR = cat(2, negBR, newNegBR(:,newConstrIdxs));
                fprintf('    nNewConstr: %d, total: %d\n', nNewConstr, size(negBR,2));
            end
        end

        
        % D: d*n matrix
        % lb: n*1 vector of 1 or -1 for label
        % Solve the linear programming:
        %   min_{s, b} \sum_i C_i*xi_i
        %          st. lb(i)*(s'*D(:,i) + b) >= 1 - xi_i
        %              xi_i >= 0
        %              s_1 >= s_2 >= ... >= s_d
        %              sum_i sw_i*s_i == 1
        function [s, b, objVal, xi] = update_s(D, lb, C, avgCosineVal)
            [d, n] = size(D);
            
            % variable x = [s; b; xi] = [s_1; ...; s_d; b; xi_1; ...; xi_n];
            cplex = Cplex('IRSVM-updateS');
            cplex.Model.sense = 'minimize';
            
            d1  = d+1;  
            d1n = d1+n; % number of variables
            
            % Add linear part of the objective
            obj = zeros(d1n,1);
            obj(d1+1:d1n) = C;
            
            % Add to the objective: sum_i C_i*xi_i
            % cplex.Model.obj = obj; % this syntax doesn't work, use the below
            cplex.addCols(obj);
                                                
            % Add constraint: s_k >= 0 and xi_i >= 0
            lbound = zeros(d1n,1);
            lbound(d1) = -inf; % no lower bound for bias b
                        
            cplex.Model.lb = lbound;
            
            % Add constraint that lb(i)*(s'*D(:,i) + b) >= 1 - xi_i
            constrVecs = D'.*repmat(lb, 1, d);
            constrVecs = sparse([constrVecs, lb, eye(n)]);
            cplex.addRows(ones(n,1), constrVecs, inf(n,1));
            
            % Add constraint that sum_i sw(i)*s_i == 1            
            constrVec = zeros(1, d1n);
%             constrVec(1:d) = 1;
%             cplex.addRows(-inf, sparse(constrVec), 1);                        
            sw = 1:d;
            sw = sqrt(sw.*(1 + (sw-1)*avgCosineVal));
            sw = [1, diff(sw)];
            constrVec(1:d) = sw;
            cplex.addRows(1, sparse(constrVec), 1);
            
                    
            % Add constraint: s_1 >= s_2 >= ... >= s_d       
            constrVecs = zeros(d-1, d1n);
            linIdx = sub2ind([d-1, d1n], 1:(d-1), 1:(d-1));
            constrVecs(linIdx) = 1;
            constrVecs(linIdx + (d-1)) = -1;
            cplex.addRows(zeros(d-1,1), constrVecs, inf(d-1,1));
            
            % Callback function for display
            cplex.DisplayFunc = []; %@disp;
                        
            % solve
            cplex.Param.threads.Cur = 1; % use a single thread
            cplex.solve();
            x = cplex.Solution.x;    
            objVal = cplex.Solution.objval;
            s = x(1:d);
            b = x(d1);
            xi = x(d1+1:end);
        end

        % A deterministic sampling method, that cycles through the sorted list of scores
        % score: n*1 vector
        % m: number of points to sample        
        function [sampleIdxs, sampleScore] = scoreSample(score, m)
            score = score(:);
            n = length(score);
            [~, sortedIdxs] = sort(score, 'descend');
            
            q = floor(m/n); % number of full rounds
            r = m - n*q; % remainder
            
            sampleIdxs = [];
            if q > 0
                sampleIdxs = repmat(sortedIdxs, q, 1); 
            end;
            sampleIdxs = [sampleIdxs; sortedIdxs(1:r)];            
            
            sampleScore = score(sampleIdxs);
            [sampleScore, sortedIdxs] = sort(sampleScore, 'descend');
            sampleIdxs = sampleIdxs(sortedIdxs);
        end;

                
        % prediction
        % Bs: 1*n cell structure for bags of instances. Bs{i} is a d*m_i matrix for m_i instances
        % w: weight vector of SVM, b: bias term
        % s: distribution weight vecor of non-increasding order        
        function score = predict(Bs, w, b, s)
            m = length(s);
            n = length(Bs);
            score = zeros(n, 1);
            for i=1:n
                score_i = Bs{i}'*w;
                [~, score_i2] = M_IRSVM.scoreSample(score_i, m);                                
                score(i) = score_i2'*s;
            end;            
            score = score + b;
        end
        
        % compute objective value from representative items
        function objVal = cmpSvmObj(D, lb, slackGrp, C, w, b)
            svmScore = D'*w + b; % before adding the bias term
            xi = max(1 - lb.*svmScore, 0);
            Cxi = C.*xi;
            nonSlgrMems = setdiff(1:length(lb), cat(2, slackGrp{:}));
            slgrCxi = zeros(1, length(slackGrp));
            for i=1:length(slackGrp)
                slgrCxi(i) = max(Cxi(slackGrp{i}));
            end;
            objVal = 0.5*(w'*w) + sum(Cxi(nonSlgrMems)) + sum(slgrCxi);
        end;
        
        % compute the objective value
        function [objVal, xi] = cmpObj(Bs, lb, C, w, b, s)
            n = length(Bs);
            m = length(s);
            score = zeros(n, 1);
            for i=1:n
                score_i = Bs{i}'*w;                 
                [~, score_i2] = M_IRSVM.scoreSample(score_i, m);                
                score(i) = score_i2'*s;                    
            end;
            score = score + b;
            xi = max(0, 1 - lb.*score);
            objVal = 0.5*(w'*w) + sum(C.*xi);            
        end
        
        
        % see linearSvm for explanation of inputs
        % D: d*n matrix
        % lb: n*1 vector
        % slackMap: n*1 vector, number from 1 to k for k slack variables
        % C: a scalar or k*1 vector
        function [w, b, objVal, xi] = linearSvm_primal(D, lb, slackMap, C)
            [d, n] = size(D);
            k = max(slackMap);
            
            % variable x = [w; b; xi] = [w_1; ...; w_d; b; xi_1; ...; xi_k];
            cplex = Cplex('pSVM.linearSVM_primal');
            cplex.Model.sense = 'minimize';
            
            d1  = d+1;
            d1k = d1+k; % number of variables
            
            % Add linear part of the objective
            % Add to the objective: sum_i C_i*xi_i
            obj = zeros(d1k,1);
            obj(d1+1:d1k) = C;
            % cplex.Model.obj = obj; % this syntax doesn't work, use the below
            cplex.addCols(obj);
            
            
            % Add to objective: 0.5*||w||^2;
            Q = zeros(d1k, d1k); % allocate d non-zero entry for Q
            for j=1:d                
                Q(j,j) = 1;                
            end            
            cplex.Model.Q = sparse(Q);
                        
            % Add constraint: xi_i >= 0                                    
            lbound = -inf(d1k,1);
            lbound(d1+1:d1k) = 0;
            cplex.Model.lb = lbound;
            
            % Add constraint that lb(i)*(w'*D(:,i) + b) >= 1 - xi_{slackMap(i)}            
            constrVecs = D'.*repmat(lb, 1, d);
            SM = zeros(n, k);
            linIdx = sub2ind([n,k], 1:n, slackMap);
            SM(linIdx) = 1;            
            constrVecs = sparse([constrVecs, lb, SM]);
            cplex.addRows(ones(n,1), constrVecs, inf(n,1));
                        
            % Callback function for display
            cplex.DisplayFunc = []; %@disp;
                        
            % solve
            cplex.Param.threads.Cur = 1; % use a single thread
            cplex.solve();
            x = cplex.Solution.x;    
            objVal = cplex.Solution.objval;
            w = x(1:d);
            b = x(d1);
            xi = x(d1+1:end);            
        end
        
        % D:  d*n data matrix for n data points
        % lb: n*1 label vector of 1, -1
        % slackGrp: a 1*m cell structure for m group of data ids.
        %   slackGrp{i} is the ids of data points that share the same slack variable        
        %   Positive data and negative data points can share the same slack variable
        %   For computation reason, it is better not to have group of with a single id        
        %   A data point does not have to be in a group (ie., a single slack for its own).
        %   A data point cannot belong two two groups
        %       e.g., slackGrp = {}, all data points have separate slack
        %       e.g., slackGrp = {[2,3,10]}, points 2, 3, 10 have the same slack
        % C: either a positive scalar or a n*1 positive vector
        %   C is the tradeoff for large margin and less constraint violation
        %   If C is n*1 vector, C(i) is associated with the tradeoff value for i^th data point
        %   Data points have the same slack variables must have the same value of C
        % Let IndSlacks = ids that do not belong to any of slackGrp{j}        
        % This function optimizes
        %   min_{w} 0.5*||w||^2 + sum_{i in IndSlacks} C_i*alpha_i 
        %                       + sum_{slack group j} C_grp_j*beta_j
        %    s.t   y_i*(w'*x_i) + b >= 1 - alpha_i for in in IndSlacks
        %          y_i*(w'*x_i) + b >= 1 - beta_j for all i in slackGrp{j}.
        %          alpha_i >= 0, beta_j >= 0
        function [w, b, alphas, dualObj, primalObj, xi] = linearSvm(D, lb, slackGrp, C)
            K = D'*D;
            maxTrial = 30;
            n = length(lb);
            for trialNo=1:maxTrial
                % although K is semidefinite, cplex might complain that it is not because of 
                % finite precision. In this case, we permute D and rerun
                try                                      
                    if trialNo ==1                        
                        [alphas, dualObj] = ML_ShareSlackSvm.kernelSvm_cplex(K, lb, slackGrp, C, [], 0, 1);
                    else
                        perIdxs = randperm(n);                                                
                        revIdxs = zeros(1, n); % for reversed mapping                        
                        revIdxs(perIdxs) = 1:n;
                        
                        % need to permute slackGrp too
                        slackGrp2 = cell(1, length(slackGrp));
                        for i=1:length(slackGrp)
                            slackGrp2{i} = revIdxs(slackGrp{i});
                        end;
               
                        [alphas2, dualObj] = ML_ShareSlackSvm.kernelSvm_cplex(K(perIdxs,perIdxs), lb(perIdxs), slackGrp2, C, [], 0, 1);
                        alphas = alphas2(revIdxs); % reverse the permutation                        
                    end
                    break; % if no error, break
                catch me   
                    fprintf('error: %s, retrying\n', me.identifier);
                end
            end
            
            w = D*(alphas.*lb);            
            
%             % In theory, there are several ways to determine b, using xi, or using the max/min score
%             % of the SVs. However, in practice, these methods are not robust, and there might be big
%             % duality gap. This perhaps is due to numerical problem. 
%             % We should use numerical optimization instead, to search for b.  
                                    
            [slackMap, k, C2] = M_IRSVM.slackGrp2slackMap(slackGrp, n, C);
            
            % variable x = [s; b; xi] = [s_1; ...; s_d; b; xi_1; ...; xi_k];
            cplex = Cplex('solve4b');
            cplex.Model.sense = 'minimize';
                        
            k1 = 1+k; % number of variables, xi and b
            
            % Add linear part of the objective
            % Add to the objective: sum_i C_i*xi_i
            obj = zeros(k1,1);
            obj(1:k) = C2;
            
            % cplex.Model.obj = obj; % this syntax doesn't work, use the below
            cplex.addCols(obj);
            
                        
            % Add constraint: xi_i >= 0                                    
            lbound = zeros(k1,1);
            lbound(end) = -inf;
            cplex.Model.lb = lbound;
            
            % Add constraint that lb(i)*(rawSvmScore(i) + b) >= 1 - xi_{slackMap(i)}            
            SM = zeros(n, k);
            linIdx = sub2ind([n,k], 1:n, slackMap);
            SM(linIdx) = 1;            
            constrVecs = sparse([SM, lb]);
            rawSvmScore = D'*w; % before adding the bias term
            cplex.addRows(ones(n,1) - lb.*rawSvmScore, constrVecs, inf(n,1));
                        
            % Callback function for display
            cplex.DisplayFunc = []; %@disp;
                        
            % solve
            cplex.Param.threads.Cur = 1; % use a single thread
            cplex.solve();
            x = cplex.Solution.x;               
            b = x(end);
            primalObj = cplex.Solution.objval + 0.5*(w'*w);
            
            svmScore = rawSvmScore + b;
            xi = max(1 - lb.*svmScore, 0);
                                    
            fprintf('nTrial: %d, dual: %g, primal: %g\n', trialNo, dualObj, primalObj);

            if primalObj - dualObj > 0.02                
                error('Duality gap is too big, something is wrong');
            end;
        end;
        
        % from slackGrp to slackMap
        % slackGrp: groups of slack variables, entries between 1 and n
        % C: a scalar or n*1 vector, the weight for slacks
        % slackMap: n*1 vector, entries between 1 and nSlack
        % C2: nSlack*1 vector for slack weights
        function [slackMap, nSlack, C2] = slackGrp2slackMap(slackGrp, n, C)
            m = length(slackGrp);
            nonSlgrMems = setdiff(1:n, cat(2, slackGrp{:}));
            k = length(nonSlgrMems);
            nSlack = k+m;

            slackMap = zeros(1, n);
            slackMap(nonSlgrMems) = 1:k;
            
            for i=1:m
                slackMap(slackGrp{i}) = k + i;
            end;            

            
            if isscalar(C)
                C2 = C;
            else
                C2 = zeros(nSlack, 1);
                C2(1:k) = C(nonSlgrMems);
                for i=1:m
                    C2(k+i) = C(slackGrp{i}(1));
                end
            end
        end        
    end    
end

