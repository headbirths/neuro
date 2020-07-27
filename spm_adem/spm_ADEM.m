

% Post-processing...
% 1007  sort  dump_dem_202006262347.txt > sorted_202006262347.txt
% 1011  cat sorted_202006262347.txt | awk 'BEGIN{s=l=v=""}{if(($1!=s)||($NF!=v)){print l; s=$1; v=$NF; l=$0}}END{print l}' > changes_202006262347.txt
% Filter out zeroes and empties
% 1019  cat changes_202006262347.txt | awk '{if(($NF!="0")&&($NF!="[]")){print}}' > nozero_202006262347.txt
% 1031  cat nozero_202006262347.txt | grep -v 'dE.dp\|dE.du\|df.dx\|dFdp\|dFdu\|Dfdx' | more




function [DEM] = spm_ADEM(DEM)

% Headbirths: DEM is both the input and the output


% Dynamic expectation maximisation:  Active inversion
% FORMAT DEM   = spm_ADEM(DEM)
%
% DEM.G  - generative process
% DEM.M  - recognition  model
% DEM.C  - causes
% DEM.U  - prior expectation of causes
%__________________________________________________________________________
%
% This implementation of DEM is the same as spm_DEM but integrates both the
% generative process and model inversion in parallel. Its functionality is 
% exactly the same apart from the fact that confounds are not accommodated
% explicitly.  The generative model is specified by DEM.G and the veridical
% causes by DEM.C; these may or may not be used as priors on the causes for
% the inversion model DEM.M (i.e., DEM.U = DEM.C).  Clearly, DEM.G does not
% require any priors or precision components; it will use the values of the
% parameters specified in the prior expectation fields.
%
% This routine is not used for model inversion per se but to simulate the
% dynamical inversion of models.  Critically, it includes action 
% variables a - that couple the model back to the generative process 
% This enables active inference (c.f., action-perception) or embodied 
% inference.
%
% hierarchical models M(i)
%--------------------------------------------------------------------------
%   M(i).g  = y(t)  = g(x,v,P)    {inline function, string or m-file}
%   M(i).f  = dx/dt = f(x,v,P)    {inline function, string or m-file}
%
%   M(i).hE = prior expectation of h hyper-parameters (cause noise)
%   M(i).hC = prior covariances of h hyper-parameters (cause noise)
%   M(i).gE = prior expectation of g hyper-parameters (state noise)
%   M(i).gC = prior covariances of g hyper-parameters (state noise)
%   M(i).Q  = precision components (input noise)
%   M(i).R  = precision components (state noise)
%   M(i).V  = fixed precision (input noise)
%   M(i).W  = fixed precision (state noise)
%   M(i).xP = precision (states)
%
%   M(i).m  = number of inputs v(i + 1);
%   M(i).n  = number of states x(i)
%   M(i).l  = number of output v(i)
%   M(i).k  = number of action a(i)

% hierarchical process G(i)
%--------------------------------------------------------------------------
%   G(i).g  = y(t)  = g(x,v,a,P)    {inline function, string or m-file}
%   G(i).f  = dx/dt = f(x,v,a,P)    {inline function, string or m-file}
%
%   G(i).pE = model-parameters
%   G(i).U  = precision (action)
%   G(i).V  = precision (input noise)
%   G(i).W  = precision (state noise)
%
%   G(1).R  = restriction or rate matrix for action [default: 1];
%   G(i).aP = precision (action)   [default: exp(-2)]
%
%   G(i).m  = number of inputs v(i + 1);
%   G(i).n  = number of states x(i)
%   G(i).l  = number of output v(i)
%   G(i).k  = number of action a(i)
%
%
% Returns the following fields of DEM
%--------------------------------------------------------------------------
%
% true model-states - u
%--------------------------------------------------------------------------
%   pU.x    = true hidden states
%   pU.v    = true causal states v{1} = response (Y)
%   pU.C    = prior covariance: cov(v)
%   pU.S    = prior covariance: cov(x)
%
% model-parameters - p
%--------------------------------------------------------------------------
%   pP.P    = parameters for each level
%
% hyper-parameters (log-transformed) - h,g
%--------------------------------------------------------------------------
%   pH.h    = cause noise
%   pH.g    = state noise
%
% conditional moments of model-states - q(u)
%--------------------------------------------------------------------------
%   qU.a    = Action
%   qU.x    = Conditional expectation of hidden states
%   qU.v    = Conditional expectation of causal states
%   qU.z    = Conditional prediction errors (v)
%   qU.C    = Conditional covariance: cov(v)
%   qU.S    = Conditional covariance: cov(x)
%
% conditional moments of model-parameters - q(p)
%--------------------------------------------------------------------------
%   qP.P    = Conditional expectation
%   qP.C    = Conditional covariance
%
% conditional moments of hyper-parameters (log-transformed) - q(h)
%--------------------------------------------------------------------------
%   qH.h    = Conditional expectation (cause noise)
%   qH.g    = Conditional expectation (state noise)
%   qH.C    = Conditional covariance
%
% F         = log evidence = log marginal likelihood = negative free energy
%__________________________________________________________________________
%
% spm_ADEM implements a variational Bayes (VB) scheme under the Laplace
% approximation to the conditional densities of states (u), parameters (p)
% and hyperparameters (h) of any analytic nonlinear hierarchical dynamic
% model, with additive Gaussian innovations.  It comprises three
% variational steps (D,E and M) that update the conditional moments of u, p
% and h respectively
%
%                D: qu.u = max <L>q(p,h)
%                E: qp.p = max <L>q(u,h)
%                M: qh.h = max <L>q(u,p)
%
% where qu.u corresponds to the conditional expectation of hidden states x
% and causal states v and so on.  L is the ln p(y,u,p,h|M) under the model
% M. The conditional covariances obtain analytically from the curvature of
% L with respect to u, p and h.
%
% The D-step is embedded in the E-step because q(u) changes with each
% sequential observation.  The dynamical model is transformed into a static
% model using temporal derivatives at each time point.  Continuity of the
% conditional trajectories q(u,t) is assured by a continuous ascent of F(t)
% in generalised co-ordinates.  This means DEM can deconvolve online and 
% represents an alternative to Kalman filtering or alternative Bayesian
% update procedures.
%
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston
% $Id: spm_ADEM.m 7679 2019-10-24 15:54:07Z spm $

fprintf('Headbirths,Header,ADEM,DynamicExpectationMaximizationAlgorithmWithActiveInversion\n' );
DUMPLOG = 1; % Set to 1 to log values that get plotted
fprintf("BEGIN,DUMP_DEM,\n");

filename = sprintf('dump_dem_%s.txt', datestr(now,'yyyymmddHHMM'));
fid = fopen(filename, 'w');
fid_to_close = fid;
%fid =1;
did = '00 ' ;
fprintf(fid, "<BEGIN> @%s\n", did);
 
% check model, data, priors and unpack
%--------------------------------------------------------------------------
DEM   = spm_ADEM_set(DEM);
M     = DEM.M;
G     = DEM.G;
C     = DEM.C;
U     = DEM.U;

% check whether to print 
%--------------------------------------------------------------------------
try
    db = DEM.db;
catch
    db = 1;
end

% find or create a DEM figure
%--------------------------------------------------------------------------
if db
    Fdem = spm_figure('GetWin','DEM');
end
 
% ensure embedding dimensions are compatible
%--------------------------------------------------------------------------
G(1).E.n = M(1).E.n;
G(1).E.d = M(1).E.n;
 

% order parameters (d = n = 1 for static models) and checks
%==========================================================================
d    = M(1).E.d + 1;                      % embedding order of q(v)
n    = M(1).E.n + 1;                      % embedding order of q(x)
s    = M(1).E.s;                          % smoothness - s.d. (bins)


% number of states and parameters - generative model
%--------------------------------------------------------------------------
nY   = size(C,2);                         % number of samples
nl   = size(M,2);                         % number of levels
nv   = sum(spm_vec(M.m));                 % number of v (causal states)
nx   = sum(spm_vec(M.n));                 % number of x (hidden states)
ny   = M(1).l;                            % number of y (inputs)
nc   = M(end).l;                          % number of c (prior causes)
nu   = nv*d + nx*n;                       % number of generalised states
 
% number of states and parameters - generative process
%--------------------------------------------------------------------------
gr   = sum(spm_vec(G.l));                 % number of v (outputs)
ga   = sum(spm_vec(G.k));                 % number of a (active states)
gx   = sum(spm_vec(G.n));                 % number of x (hidden states)
gy   = G(1).l;                            % number of y (inputs)
na   = ga;                                % number of a (action)
 
% number of iterations
%--------------------------------------------------------------------------
try, nE = M(1).E.nE; catch, nE = 16; end
try, nM = M(1).E.nM; catch, nM = 8;  end
try, dt = M(1).E.dt; catch, dt = 1;  end
 
 
% initialise regularisation parameters
%--------------------------------------------------------------------------
te = 2;                                   % log integration time for E-Step
global t


% precision (roughness) of generalised fluctuations
%--------------------------------------------------------------------------
iV    = spm_DEM_R(n,s); % returns the precision of the temporal derivatives of a Gaussian process % FORMAT [R,V] = spm_DEM_R(n,s,form)
iG    = spm_DEM_R(n,s); % returns the precision of the temporal derivatives of a Gaussian process % FORMAT [R,V] = spm_DEM_R(n,s,form)

% time-delay operators (absorb motor delays into motor gain matrix)
%--------------------------------------------------------------------------
try
    nG = norm(iG);
    iG = iG*spm_DEM_T(n,-M(1).Ta);  
    iG = iG*nG/norm(iG);
end
% spm_DEM_T =Create a [template] model structure
% FORMAT [M] = spm_DEM_M(model,l,n)
% FORMAT [M] = spm_DEM_M(model,X1,X2,...)
% model: 'General linear model','GLM'            Headbirths!!! But 'n' is not one of these!
%        'Factor analysis','FA'
%        'Independent component analysis','ICA'
%        'Sparse coding',"SC'
%        'convolution model'
%        'State space model','SSM',','Double Well'
%        'Lorenz'
%        'Ornstein_Uhlenbeck','OU'

try
    Ty = spm_DEM_T(n,-M(1).Ty); % Headbirths!!! But 'n' is not one of these!
    Ty = kron(Ty,speye(ny,ny)); % Kronecker tensor product 
end
 
% precision components Q{} requiring [Re]ML estimators (M-Step)
%==========================================================================
Q     = {};
for i = 1:nl
    q0{i,i} = sparse(M(i).l,M(i).l); %#ok<AGROW>
    r0{i,i} = sparse(M(i).n,M(i).n);
end
Q0    = kron(iV,spm_cat(q0));
R0    = kron(iV,spm_cat(r0));
for i = 1:nl
    for j = 1:length(M(i).Q)
        q          = q0;
        q{i,i}     = M(i).Q{j};
        Q{end + 1} = blkdiag(kron(iV,spm_cat(q)),R0);
    end
    for j = 1:length(M(i).R)
        q          = r0;
        q{i,i}     = M(i).R{j};
        Q{end + 1} = blkdiag(Q0,kron(iV,spm_cat(q)));
    end
end
 
 
% and fixed components P
%--------------------------------------------------------------------------
Q0    = kron(iV,spm_cat(spm_diag({M.V})));
R0    = kron(iV,spm_cat(spm_diag({M.W})));
Qp    = blkdiag(Q0,R0);
nh    = length(Q);                           % number of hyperparameters
iR    = [zeros(1,ny),ones(1,nv),ones(1,nx)]; % for empirical priors
iR    = kron(speye(n,n),diag(iR)); 

% restriction or rate matrices - in terms of precision
%--------------------------------------------------------------------------
q0{1} = G(1).U;
Q0    = kron(iG,spm_cat(q0));
R0    = kron(iG,spm_cat(r0));
iG    = blkdiag(Q0,R0);

% restriction or rate matrices - in terms of dE/da
%--------------------------------------------------------------------------
try
    R         = sparse(sum(spm_vec(G.l)),na);
    R(1:ny,:) = G(1).R;
    R         = kron(spm_speye(n,1,0),R);
catch
    R = 1;
end

% fixed priors on action (a)
%--------------------------------------------------------------------------
try
    aP = G(1).aP;
catch
    aP = exp(-2);
end

% fixed priors on states (u)
%--------------------------------------------------------------------------
xP    = spm_cat(spm_diag({M.xP}));
Px    = kron(iV(1:n,1:n),speye(nx,nx)*exp(-8) + xP);
Pv    = kron(iV(1:d,1:d),speye(nv,nv)*exp(-8));
Pa    = spm_speye(na,na)*aP;
Pu    = spm_cat(spm_diag({Px Pv}));
 
% hyperpriors
%--------------------------------------------------------------------------
ph.h  = spm_vec({M.hE M.gE});             % prior expectation of h
ph.c  = spm_cat(spm_diag({M.hC M.gC}));   % prior covariances of h
qh.h  = ph.h;                             % conditional expectation
qh.c  = ph.c;                             % conditional covariance
ph.ic = spm_inv(ph.c);                    % prior precision
 
% priors on parameters (in reduced parameter space)
%==========================================================================
pp.c  = cell(nl,nl);
qp.p  = cell(nl,1);
for i = 1:(nl - 1)
 
    % eigenvector reduction: p <- pE + qp.u*qp.p
    %----------------------------------------------------------------------
    qp.u{i}   = spm_svd(M(i).pC);                    % basis for parameters
    M(i).p    = size(qp.u{i},2);                     % number of qp.p
    qp.p{i}   = sparse(M(i).p,1);                    % initial qp.p
    pp.c{i,i} = qp.u{i}'*M(i).pC*qp.u{i};            % prior covariance
    
    try
        qp.e{i} = qp.p{i} + qp.u{i}'*(spm_vec(M(i).P) - spm_vec(M(i).pE));
    catch
        qp.e{i} = qp.p{i};                           % initial qp.e
    end
 
end
Up    = spm_cat(spm_diag(qp.u));
 
% initialise and augment with confound parameters B; with flat priors
%--------------------------------------------------------------------------
np    = sum(spm_vec(M.p));                  % number of model parameters
pp.c  = spm_cat(pp.c);
pp.ic = spm_inv(pp.c);
 
% initialise conditional density q(p) (for D-Step)
%--------------------------------------------------------------------------
qp.e  = spm_vec(qp.e);
qp.c  = sparse(np,np);
 
% initialise cell arrays for D-Step; e{i + 1} = (d/dt)^i[e] = e[i]
%==========================================================================
qu.x      = cell(n,1);
qu.v      = cell(n,1);
qu.a      = cell(1,1);
qu.y      = cell(n,1);
qu.u      = cell(n,1);
pu.v      = cell(n,1);
pu.x      = cell(n,1);
pu.z      = cell(n,1);
pu.w      = cell(n,1);
 
[qu.x{:}] = deal(sparse(nx,1));
[qu.v{:}] = deal(sparse(nv,1));
[qu.a{:}] = deal(sparse(na,1));
[qu.y{:}] = deal(sparse(ny,1));
[qu.u{:}] = deal(sparse(nc,1));
[pu.v{:}] = deal(sparse(gr,1));
[pu.x{:}] = deal(sparse(gx,1));
[pu.z{:}] = deal(sparse(gr,1));
[pu.w{:}] = deal(sparse(gx,1));
 
% initialise cell arrays for hierarchical structure of x[0] and v[0]
%--------------------------------------------------------------------------
qu.x{1}   = spm_vec({M(1:end - 1).x});
qu.v{1}   = spm_vec({M(1 + 1:end).v});
qu.a{1}   = spm_vec({G.a});
pu.x{1}   = spm_vec({G.x});
pu.v{1}   = spm_vec({G.v});
 
 
% derivatives for Jacobian of D-step
%--------------------------------------------------------------------------
Dx    = kron(spm_speye(n,n,1),spm_speye(nx,nx,0));
Dv    = kron(spm_speye(d,d,1),spm_speye(nv,nv,0));
Dc    = kron(spm_speye(d,d,1),spm_speye(nc,nc,0));
Da    = kron(spm_speye(1,1,1),sparse(na,na));
Du    = spm_cat(spm_diag({Dx,Dv}));
Dq    = spm_cat(spm_diag({Dx,Dv,Dc,Da}));
 
Dx    = kron(spm_speye(n,n,1),spm_speye(gx,gx,0));
Dv    = kron(spm_speye(n,n,1),spm_speye(gr,gr,0));
Dp    = spm_cat(spm_diag({Dv,Dx,Dv,Dx}));
dfdw  = kron(speye(n,n),speye(gx,gx));
dydv  = kron(speye(n,n),speye(gy,gr));
 
% and null blocks
%--------------------------------------------------------------------------
dVdc  = sparse(d*nc,1);
 
% gradients and curvatures for conditional uncertainty
%--------------------------------------------------------------------------
dWdu  = sparse(nu,1);
dWduu = sparse(nu,nu);
 
% preclude unnecessary iterations
%--------------------------------------------------------------------------
if ~np && ~nh, nE = 1; end
 
 
% create innovations (and add causes)
%--------------------------------------------------------------------------
[z,w]  = spm_DEM_z(G,nY);
z{end} = C + z{end};
a      = {G.a};
Z      = spm_cat(z(:));
W      = spm_cat(w(:));
A      = spm_cat(a(:));
 
% Iterate DEM
%==========================================================================
F      = -Inf;
for iE = 1:nE
    fprintf('Headbirths,Waypoint,DEMit%d\n' , iE);
    did = sprintf( '%02d', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
    fprintf('Headbirths,DumpIteration,%s\n' , did);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DUMP PER ITERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Dumping is at the *start* of the iteration: we get to see the initialization state; 
    % The final state should be the same as the penultimate if the number of iterations is sufficient to achieve equilibrium
    if 1
        fprintf('Headbirths,DumpStart,%s\n' , did);
        for i=1:length(M); % [1×2 struct] % generative model
            prefix = sprintf('DEM.M(%d)', i);
            %M=DEM.M;
            %E=DEM.M(i).E;
            if(isempty(M(i).E))
                fprintf(fid, did, "DEM.M(%d).E = []; %%$ \n", i);
            else
                disp_scalar(fid, did, strcat(prefix, ".E.s "), M(i).E.s  )    % (DEM.M(i).E.s )     : [1×1 struct] 0.5000
                disp_scalar(fid, did, strcat(prefix, ".E.n "), M(i).E.n  )    % (DEM.M(i).E.n )     : 4
                disp_scalar(fid, did, strcat(prefix, ".E.d "), M(i).E.d  )    % (DEM.M(i).E.d )     : 2
                disp_scalar(fid, did, strcat(prefix, ".E.dt"), M(i).E.dt )    % (DEM.M(i).E.dt )    : 1  Headbirths delta-t for Euler discrete simulation of differential equations
                disp_scalar(fid, did, strcat(prefix, ".E.nE"), M(i).E.nE )    % (DEM.M(i).E.nE )    : 16 Headbirths number of Expectation phase iterations
                disp_scalar(fid, did, strcat(prefix, ".E.nD"), M(i).E.nD )    % (DEM.M(i).E.nD )    : 1
                disp_scalar(fid, did, strcat(prefix, ".E.nM"), M(i).E.nM )    % (DEM.M(i).E.nM )    : 8  Headbirths number of Maximization phase iterations
                disp_scalar(fid, did, strcat(prefix, ".E.nN"), M(i).E.nN )    % (DEM.M(i).E.nN )    : 8
                disp_scalar(fid, did, strcat(prefix, ".E.linear"), M(i).E.linear); % (DEM.M(i).E.linear) : 0
            end
            disp_vec(fid, did, strcat(prefix, ".x"), M(i).x ); % : [2×1 double]
            % FUNCTION % disp_scalar(fid, did, strcat(prefix, ".f"), DEM.M(i).f ); % : @spm_fx_mountaincar  M(i).g  = y(t)  = g(x,v,P)    {inline function, string or m-file}
            % FUNCTION % disp_scalar(fid, did, strcat(prefix, ".g"), DEM.M(i).g ); % : @(x,v,P)x            M(i).f  = dx/dt = f(x,v,P)    {inline function, string or m-file}
            if(isempty(DEM.M(i).pE )) % : [1×1 struct]) M(i).pE = prior expectation of p model-parameters
                fprintf(fid, did, "DEM.M(%d).pE = []; %%$ \n", i);
            else
                disp_scalar(fid, did, strcat(prefix, ".pE.a"), M(i).pE.a ); % : [1×1 struct]
                disp_vec(fid,    did, strcat(prefix, ".pE.b"), M(i).pE.b ); % : [1×1 struct]
                disp_vec(fid,    did, strcat(prefix, ".pE.c"), M(i).pE.c ); % : [1×1 struct]
                disp_scalar(fid, did, strcat(prefix, ".pE.d"), M(i).pE.d ); % : [1×1 struct]
            end
            disp_array(fid, did, strcat(prefix, ".pC"), full(M(i).pC) ); % : [8×8 double] M(i).pC = prior covariances of p model-parameters
            disp_array(fid, did, strcat(prefix, ".V"), full(M(i).V) ); % : [2×2 double] M(i).V  = fixed precision (input noise)
            disp_array(fid, did, strcat(prefix, ".W"), full(M(i).W) ); % : [2×2 double] M(i).W  = fixed precision (state noise)
            disp_scalar(fid, did, strcat(prefix, ".a"), full(M(i).a) ); % : []
            disp_vec(fid, did, strcat(prefix, ".v"), full(M(i).v) ); % : [2×1 double]
            disp_scalar(fid, did, strcat(prefix, ".m"), M(i).m ); % M(i).m  = number of inputs v(i + 1)
            disp_scalar(fid, did, strcat(prefix, ".n"), M(i).n ); % M(i).n  = number of states x(i)
            disp_scalar(fid, did, strcat(prefix, ".l"), M(i).l ); % M(i).l  = number of output v(i)
            disp_scalar(fid, did, strcat(prefix, ".k"), M(i).k ); % M(i).k  = number of action a(i)
            disp_array(fid, did, strcat(prefix, ".U"), full(M(i).U) ); % : [2×2 double] % [exogenous] causes
            disp_scalar(fid, did, strcat(prefix, ".fq"), M(i).fq ); % : 'spm_mountaincar_Q'
            disp_array(fid, did, strcat(prefix, ".X"),  full(M(i).X) ); % : [1024×2 double]
            disp_array(fid, did, strcat(prefix, ".xP"), full(M(i).xP) ); % : [2×2 double] M(i).xP = precision (states)
            disp_array(fid, did, strcat(prefix, ".vP"), full(M(i).vP) ); % : [2×2 double]
            disp_scalar(fid, did, strcat(prefix, ".Q "), M(i).Q  ); % : []  M(i).Q  = precision components (input noise)
            disp_scalar(fid, did, strcat(prefix, ".R "), M(i).R  ); % : []  M(i).R  = precision components (state noise)
            disp_scalar(fid, did, strcat(prefix, ".hE"), M(i).hE ); % : [0×1 double] M(i).hE = prior expectation of h hyper-parameters (cause noise)
            disp_scalar(fid, did, strcat(prefix, ".gE"), M(i).gE ); % : [0×1 double] M(i).gE = prior expectation of g hyper-parameters (state noise)
            disp_scalar(fid, did, strcat(prefix, ".ph"), M(i).ph ); % : []
            disp_scalar(fid, did, strcat(prefix, ".pg"), M(i).pg ); % : []
            disp_scalar(fid, did, strcat(prefix, ".hC"), M(i).hC ); % : [] M(i).hC = prior covariances of h hyper-parameters (cause noise)
            disp_scalar(fid, did, strcat(prefix, ".gC"), M(i).gC ); % : [] M(i).gC = prior covariances of g hyper-parameters (state noise)
            disp_scalar(fid, did, strcat(prefix, ".sv"), M(i).sv ); % : 0.5000
            disp_scalar(fid, did, strcat(prefix, ".sw"), M(i).sw ); % : 0.5000
            % MISSING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
            %if(isempty(DEM.M(i).p )) % Headbirths <-- Not just empty but doesn't exist
            %    fprintf(fid, did, "DEM.M(%d).p = []; %%$ \n", i);
            %else
            %    disp_scalar(fid, did, strcat(prefix, ".p "), M(i).p  ); % : 8
            %end
        end

        % Note: At top level, this is DEM(i).G, not G
        for i=1:length(G); % [1×1 struct]
            prefix = sprintf('DEM.G(%d)', i);
            if(isempty(G(i).E )) % : [1×1 struct])
                fprintf(fid, did, "DEM.G(%d).E = []; %%$ \n", i);
            else
                disp_scalar(fid, did, strcat(prefix, ".E.s "), G(i).E.s ); % : 0.5000
                disp_scalar(fid, did, strcat(prefix, ".E.n "), G(i).E.n ); % : 4
                disp_scalar(fid, did, strcat(prefix, ".E.d "), G(i).E.d ); % : 2
                disp_scalar(fid, did, strcat(prefix, ".E.dt"), G(i).E.dt ); % : 1
            end
            disp_vec(fid, did, strcat(prefix, ".x"), G(i).x ); %   [2×1 double]
            % FUNCTION % disp_scalar(fid, did, strcat(prefix, ".f "), G(i).f  ); %   @spm_fx_mountaincar
            % FUNCTION % disp_scalar(fid, did, strcat(prefix, ".g "), G(i).g  ); %   @(x,v,a,P)x
            if(isempty(G(i).pE )) % : [1×1 struct])
                fprintf(fid, did, "DEM.G(%d).pE = []; %%$ \n", i);
            else
                disp_scalar(fid, did, strcat(prefix, ".pE.a"), G(i).pE.a ); %   [1×1 struct]
                disp_vec(fid, did, strcat(prefix, ".pE.b"), G(i).pE.b ); %   [1×1 struct]
                disp_vec(fid, did, strcat(prefix, ".pE.c"), G(i).pE.c ); %   [1×1 struct]
                disp_scalar(fid, did, strcat(prefix, ".pE.d"), G(i).pE.d ); %   [1×1 struct]
            end
            disp_array(fid, did, strcat(prefix, ".pC"), full(G(i).pC) ); %   [8×8 double]
            disp_array(fid, did, strcat(prefix, ".V"),  full(G(i).V) ); %   [2×2 double]
            disp_array(fid, did, strcat(prefix, ".W"),  full(G(i).W) ); %   [2×2 double]
            disp_vec(fid, did, strcat(prefix, ".a"), G(i).a ); %   [0×1 double]
            disp_vec(fid, did, strcat(prefix, ".v"), G(i).v ); %   [2×1 double]
            disp_scalar(fid, did, strcat(prefix, ".m"), G(i).m ); %   1
            disp_scalar(fid, did, strcat(prefix, ".n"), G(i).n ); %   2
            disp_scalar(fid, did, strcat(prefix, ".l"), G(i).l ); %   2
            disp_scalar(fid, did, strcat(prefix, ".k"), G(i).k ); %   0
            disp_array(fid, did, strcat(prefix, ".U"), full(G(i).U) ); %   [2×2 double]
            disp_scalar(fid, did, strcat(prefix, ".sv"), G(i).sv ); %   0.5000
            disp_scalar(fid, did, strcat(prefix, ".sw"), G(i).sw ); %   0.5000
            disp_scalar(fid, did, strcat(prefix, ".fq"), G(i).fq ); %   'spm_mountaincar_Q'
            disp_array(fid, did, strcat(prefix, ".X"), full(G(i).X) ); %   [1024×2 double]
        end

        disp_vec(fid, did, "DEM.C", full(C))      % [1×128 double]
        disp_vec(fid, did, "DEM.V", full(U))      % [1×128 double]
        disp_scalar(fid, did, "DEM.class", DEM.class)  % 'active'
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END DUMP PER ITERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    
    % get time and clear persistent variables in evaluation routines
    %----------------------------------------------------------------------
    tic; clear spm_DEM_eval
 
    % E-Step: (with embedded D-Step)
    %======================================================================
 
    % [re-]set accumulators for E-Step
    %----------------------------------------------------------------------
    dFdp  = zeros(np,1);
    dFdpp = zeros(np,np);
    EE    = sparse(0);
    ECE   = sparse(0);
    EiSE  = sparse(0);
    qp.ic = sparse(0);
    Hqu.c = sparse(0);
 
 
    % [re-]set precisions using [hyper]parameter estimates
    %----------------------------------------------------------------------
    iS    = Qp;
    for i = 1:nh
       iS = iS + Q{i}*exp(qh.h(i));
    end
    did = sprintf( '%02da', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
    disp_variable(fid, did, "iS", full(iS) );

    
    % precision for empirical priors
    %----------------------------------------------------------------------
    iP    = iR*iS*iR;
    disp_variable(fid, did, "iP", full(iP) );
    
    % [re-]set states & their derivatives
    %----------------------------------------------------------------------
    try
        qu = qU(1);
        pu = pU(1);
    end

    disp_variable(fid, did, "qu", qu );
    disp_variable(fid, did, "qu", pu );
 
    % D-Step: (nY samples)
    %======================================================================
    for iY = 1:nY
        did = sprintf( '%02db', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        disp_variable(fid, did, "iY", iY ); %   0.5000
 
        % time (GLOBAL variable for non-automomous systems)
        %------------------------------------------------------------------
        t      = iY/nY;
        
        % pass action to pu.a (external states)
        %==================================================================
        try, A = spm_cat({qU.a qu.a}); end
        
        % derivatives of responses and random fluctuations
        %------------------------------------------------------------------
        pu.z = spm_DEM_embed(Z,n,iY);
        pu.w = spm_DEM_embed(W,n,iY);
        pu.a = spm_DEM_embed(A,n,iY);
        qu.u = spm_DEM_embed(U,n,iY);

        disp_variable(fid, did, "pu.z", pu.z );
        disp_variable(fid, did, "pu.w", pu.w );
        disp_variable(fid, did, "pu.a", pu.a );
        disp_variable(fid, did, "qu.u", qu.u );
        
        
        % evaluate generative process
        %------------------------------------------------------------------
        [pu,dg,df] = spm_ADEM_diff(G,pu);
        did = sprintf( '%02dc', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        disp_variable(fid, did, "pu", pu );
        disp_variable(fid, did, "dg", dg );
        disp_variable(fid, did, "df", df );
 
        
        % and pass response to qu.y
        %==================================================================
        for i = 1:n
            y       = spm_unvec(pu.v{i},{G.v});
            qu.y{i} = y{1};
        end
        disp_variable(fid, did, "qu.y", qu.y );
        
        % sensory delays
        %------------------------------------------------------------------
        try, qu.y = spm_unvec(Ty*spm_vec(qu.y),qu.y); end
        did = sprintf( '%02dd', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        disp_variable(fid, did, "qu.y", qu.y );
        
        
        % evaluate generative model
        %------------------------------------------------------------------       
        [E,dE] = spm_DEM_eval(M,qu,qp);
        disp_variable(fid, did, "E", E );
        disp_variable(fid, did, "dE", dE );

        
        % conditional covariance [of states {u}]
        %------------------------------------------------------------------
        qu.c   = spm_inv(dE.du'*iS*dE.du + Pu);
        pu.c   = spm_inv(dE.du'*iP*dE.du + Pu);
        Hqu.c  = Hqu.c + spm_logdet(qu.c);
        disp_variable(fid, did, "qu.c ", qu.c  );
        disp_variable(fid, did, "pu.c ", pu.c  );
        disp_variable(fid, did, "Hqu.c", Hqu.c );
        
        % save at qu(t)
        %------------------------------------------------------------------
        qE{iY} = E;
        qC{iY} = qu.c;
        pC{iY} = pu.c;
        qU(iY) = qu;
        pU(iY) = pu;
        disp_variable(fid, did, "iY ", iY  );
        disp_variable(fid, did, "qE{iY} ", qE{iY}  );
        disp_variable(fid, did, "qC{iY} ", qC{iY}  );
        disp_variable(fid, did, "pC{iY} ", pC{iY}  );
        disp_variable(fid, did, "qU(iY) ", qU(iY)  );
        disp_variable(fid, did, "pU(iY) ", pU(iY)  );
 
        % and conditional precision
        %------------------------------------------------------------------
        disp_variable(fid, did, "nh", nh  );
        if nh
            ECEu  = dE.du*qu.c*dE.du';
            ECEp  = dE.dp*qp.c*dE.dp';
            disp_variable(fid, did, "ECEu ", ECEu  );
            disp_variable(fid, did, "ECEp ", ECEp  );
        end
 
        
        % uncertainty about parameters dWdv, ... ; W = ln(|qp.c|)
        %==================================================================
        did = sprintf( '%02de', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        disp_variable(fid, did, "np", np  );
        if np
            disp_variable(fid, did, "nu", nu  );
            for i = 1:nu
                CJp(:,i)   = spm_vec(qp.c*dE.dpu{i}'*iS);
                dEdpu(:,i) = spm_vec(dE.dpu{i}');
            end
            dWdu  = CJp'*spm_vec(dE.dp');
            dWduu = CJp'*dEdpu;
            disp_variable(fid, did, "dWdu", dWdu   );
            disp_variable(fid, did, "dWduu ", dWduu  );
        end
        
        % tensor products for Jacobian (generative process)
        %------------------------------------------------------------------
        Dgda  = kron(spm_speye(n,1,1),dg.da);
        Dgdv  = kron(spm_speye(n,n,1),dg.dv);
        Dgdx  = kron(spm_speye(n,n,1),dg.dx);
        dfda  = kron(spm_speye(n,1,0),df.da);
        dfdv  = kron(spm_speye(n,n,0),df.dv);
        dfdx  = kron(spm_speye(n,n,0),df.dx);       
        dgda  = kron(spm_speye(n,1,0),dg.da);
        dgdx  = kron(spm_speye(n,n,0),dg.dx);

        disp_variable(fid, did, "Dgda", Dgda  );
        disp_variable(fid, did, "Dgdv", Dgdv  );
        disp_variable(fid, did, "Dgdx", Dgdx  );
        disp_variable(fid, did, "dfda", dfda  );
        disp_variable(fid, did, "dfdv", dfdv  );
        disp_variable(fid, did, "dfdx", dfdx  );
        disp_variable(fid, did, "dgda", dgda  );
        disp_variable(fid, did, "dgdx", dgdx  );
        
        % change in error w.r.t. action
        %------------------------------------------------------------------
        Dfdx  = 0;
        for i = 1:n
            Dfdx = Dfdx + kron(spm_speye(n,n,-i),df.dx^(i - 1));
        end
        did = sprintf( '%02df', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        disp_variable(fid, did, "Dfdx", Dfdx  );
        
        % dE/da with restriction (R)
        %------------------------------------------------------------------
        dE.dv = dE.dy*dydv;
        dE.da = dE.dv*((dgda + dgdx*Dfdx*dfda).*R);
        disp_variable(fid, did, "dE.dv", dE.dv  );
        disp_variable(fid, did, "dE.da", dE.da  );

        
        % first-order derivatives
        %------------------------------------------------------------------
        dVdu  = -dE.du'*iS*E - Pu*spm_vec({qu.x{1:n} qu.v{1:d}}) - dWdu/2;
        dVda  = -dE.da'*iG*E - Pa*spm_vec( qu.a{1:1});
        disp_variable(fid, did, "dVdu", dVdu  );
        disp_variable(fid, did, "dVda", dVda  );
        
        
        % and second-order derivatives
        %------------------------------------------------------------------
        dVduu = -dE.du'*iS*dE.du - Pu - dWduu/2 ;
        dVdaa = -dE.da'*iG*dE.da - Pa;
        dVduv = -dE.du'*iS*dE.dv;
        dVduc = -dE.du'*iS*dE.dc;
        dVdua = -dE.du'*iS*dE.da;
        dVdav = -dE.da'*iG*dE.dv;
        dVdau = -dE.da'*iG*dE.du;
        dVdac = -dE.da'*iG*dE.dc;
        disp_variable(fid, did, "dVduu", dVduu  );
        disp_variable(fid, did, "dVdaa", dVdaa  );
        disp_variable(fid, did, "dVduv", dVduv  );
        disp_variable(fid, did, "dVduc", dVduc  );
        disp_variable(fid, did, "dVdua", dVdua  );
        disp_variable(fid, did, "dVdav", dVdav  );
        disp_variable(fid, did, "dVdau", dVdau  );
        disp_variable(fid, did, "dVdac", dVdac  );
 
         
        % D-step update: of causes v{i}, and hidden states x(i)
        %==================================================================
 
        did = sprintf( '%02dg', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        % states and conditional modes
        %------------------------------------------------------------------
        p     = {pu.v{1:n} pu.x{1:n} pu.z{1:n} pu.w{1:n}};
        q     = {qu.x{1:n} qu.v{1:d} qu.u{1:d} qu.a{1:1}};
        u     = [p q];  
        disp_variable(fid, did, "p", p  );
        disp_variable(fid, did, "q", q  );
        disp_variable(fid, did, "u", u  );
        
        % gradient
        %------------------------------------------------------------------
        dFdu  = [                              Dp*spm_vec(p); 
                 spm_vec({dVdu; dVdc; dVda}) + Dq*spm_vec(q)];
        disp_variable(fid, did, "dFdu", dFdu  );
 
 
        % Jacobian (variational flow)
        %------------------------------------------------------------------
        dFduu = spm_cat(...
                {Dgdv  Dgdx Dv   []   []       []    Dgda;
                 dfdv  dfdx []   dfdw []       []    dfda;
                 []    []   Dv   []   []       []    [];
                 []    []   []   Dx   []       []    [];
                 dVduv []   []   []   Du+dVduu dVduc dVdua;
                 []    []   []   []   []       Dc    []
                 dVdav []   []   []   dVdau    dVdac dVdaa});
        disp_variable(fid, did, "dFduu", dFduu  );
 
 
        % update states q = {x,v,z,w} and conditional modes
        %==================================================================
        du    = spm_dx(dFduu,dFdu,dt);
        u     = spm_unvec(spm_vec(u) + du,u);
        disp_variable(fid, did, "du", du  );
        disp_variable(fid, did, "u", u  );
 
        % and save them
        %------------------------------------------------------------------
        pu.v(1:n) = u((1:n));
        pu.x(1:n) = u((1:n) + n);
        qu.x(1:n) = u((1:n) + n + n + n + n);
        qu.v(1:d) = u((1:d) + n + n + n + n + n);
        qu.a(1:1) = u((1:1) + n + n + n + n + n + d + d);
        disp_variable(fid, did, "pu.v", pu.v  );
        disp_variable(fid, did, "pu.x", pu.x  );
        disp_variable(fid, did, "qu.x", qu.x  );
        disp_variable(fid, did, "qu.v", qu.v  );
        disp_variable(fid, did, "qu.a", qu.a  );
        

        % Gradients and curvatures for E-Step: W = tr(C*J'*iS*J)
        %==================================================================
        did = sprintf( '%02dh', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        disp_variable(fid, did, "np", np  );
        if np
            for i = 1:np
                CJu(:,i)   = spm_vec(qu.c*dE.dup{i}'*iS);
                dEdup(:,i) = spm_vec(dE.dup{i}');
            end
            dWdp  = CJu'*spm_vec(dE.du');
            dWdpp = CJu'*dEdup;
            disp_variable(fid, did, "dWdp", dWdp  );
            disp_variable(fid, did, "dWdpp", dWdpp  );
 
            % Accumulate; dF/dP = <dL/dp>, dF/dpp = ...
            %--------------------------------------------------------------
            dFdp  = dFdp  - dWdp/2  - dE.dp'*iS*E;
            dFdpp = dFdpp - dWdpp/2 - dE.dp'*iS*dE.dp;
            qp.ic = qp.ic           + dE.dp'*iS*dE.dp;
            disp_variable(fid, did, "dFdp", dFdp   );
            disp_variable(fid, did, "dFdpp", dFdpp  );
            disp_variable(fid, did, "qp.ic", qp.ic  );
            
        end
 
        did = sprintf( '%02dj', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        % accumulate SSE
        %------------------------------------------------------------------
        EiSE = EiSE + E'*iS*E;
        disp_variable(fid, did, "EiSE", EiSE  );
        
        % and quantities for M-Step
        %------------------------------------------------------------------
        disp_variable(fid, did, "nh", nh  );
        if nh
            EE  = E*E'+ EE;
            ECE = ECE + ECEu + ECEp;
            disp_variable(fid, did, "EE", EE  );
            disp_variable(fid, did, "ECE", CEE  );
        end
        
        if nE == 1
          disp_variable(fid, did, "nE", nE  );
          
            % evaluate objective function (F)
            %======================================================================
            J(iY) = - trace(E'*iS*E)/2  ...            % states (u)
                    + spm_logdet(qu.c)  ...            % entropy q(u)
                    + spm_logdet(iS)/2;                % entropy - error
            disp_variable(fid, did, "iY", iY  );
            disp_variable(fid, did, "J(iY)", J(iY)  );
        end
        
    end % sequence (nY)
 
    did = sprintf( '%02dk', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)

    % augment with priors
    %----------------------------------------------------------------------
    dFdp   = dFdp  - pp.ic*qp.e;
    dFdpp  = dFdpp - pp.ic;
    qp.ic  = qp.ic + pp.ic;
    qp.c   = spm_inv(qp.ic);
    disp_variable(fid, did, "dFdp ", dFdp   );
    disp_variable(fid, did, "dFdpp", dFdpp  );
    disp_variable(fid, did, "qp.ic", qp.ic  );
    disp_variable(fid, did, "qp.c ", qp.c   );
 
 
    % E-step: update expectation (p)
    %======================================================================
 
    % update conditional expectation
    %----------------------------------------------------------------------
    dp     = spm_dx(dFdpp,dFdp,{te});
    qp.e   = qp.e + dp;
    qp.p   = spm_unvec(qp.e,qp.p);
    did = sprintf( '%02dl', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
    disp_variable(fid, did, "dp  ", dp     );
    disp_variable(fid, did, "qp.e", qp.e   );
    disp_variable(fid, did, "qp.p", qp.p   );
 
 
    % M-step - hyperparameters (h = exp(l))
    %======================================================================
    mh     = zeros(nh,1);
    dFdh   = zeros(nh,1);
    dFdhh  = zeros(nh,nh);
    for iM = 1:nM
        did = sprintf( '%02dm', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
 
        % [re-]set precisions using [hyper]parameter estimates
        %------------------------------------------------------------------
        iS    = Qp;
        for i = 1:nh
            iS = iS + Q{i}*exp(qh.h(i));
        end
        S     = spm_inv(iS);
        dS    = ECE + EE - S*nY;
        disp_variable(fid, did, "S", S    );
        disp_variable(fid, did, "dS", dS   );
         
        % 1st-order derivatives: dFdh = dF/dh
        %------------------------------------------------------------------
        for i = 1:nh
            dPdh{i}        =  Q{i}*exp(qh.h(i));
            dFdh(i,1)      = -trace(dPdh{i}*dS)/2;
        end
        % WORKAROUND!!! disp_variable(fid, did, "dPdh", dPdh   );
        % WORKAROUND!!! disp_variable(fid, did, "dFdh", dFdh   );
 
        % 2nd-order derivatives: dFdhh
        %------------------------------------------------------------------
        for i = 1:nh
            for j = 1:nh
                dFdhh(i,j) = -trace(dPdh{i}*S*dPdh{j}*S*nY)/2;
            end
        end
        % WORKAROUND!!! disp_variable(fid, did, "dFdhh", dFdhh   );
 
        % hyperpriors
        %------------------------------------------------------------------
        qh.e  = qh.h  - ph.h;
        dFdh  = dFdh  - ph.ic*qh.e;
        dFdhh = dFdhh - ph.ic;
        disp_variable(fid, did, "qh.e", qh.e    );
        % WORKAROUND!!! disp_variable(fid, did, "dFdh", dFdh    );
        % WORKAROUND!!! disp_variable(fid, did, "dFdhh", dFdhh   );
 
        % update ReML estimate of parameters
        %------------------------------------------------------------------
        dh    = spm_dx(dFdhh,dFdh);
        qh.h  = qh.h + dh;
        mh    = mh   + dh;
        disp_variable(fid, did, "dh", dh      );
        disp_variable(fid, did, "qh.h", qh.h    );
        disp_variable(fid, did, "mh", mh      );
 
        % conditional covariance of hyperparameters
        %------------------------------------------------------------------
        qh.c  = -spm_inv(dFdhh);
        disp_variable(fid, did, "qh.c", qh.c    );
 
        % convergence (M-Step)
        %------------------------------------------------------------------
        if (dFdh'*dh < 1e-2) || (norm(dh,1) < exp(-8)), break, end
 
    end % M-Step
 
    % evaluate objective function (F)
    %======================================================================
    L   = - trace(EiSE)/2  ...               % states (u)
        - trace(qp.e'*pp.ic*qp.e)/2  ...     % parameters (p)
        - trace(qh.e'*ph.ic*qh.e)/2  ...     % hyperparameters (h)
        + Hqu.c/2             ...            % entropy q(u)
        + spm_logdet(qp.c)/2  ...            % entropy q(p)
        + spm_logdet(qh.c)/2  ...            % entropy q(h)
        - spm_logdet(pp.c)/2  ...            % entropy - prior p
        - spm_logdet(ph.c)/2  ...            % entropy - prior h
        + spm_logdet(iS)*nY/2 ...            % entropy - error
        - n*ny*nY*log(2*pi)/2;

    disp_variable(fid, did, "L", L    );
    
    
    % if F is increasing, save expansion point and derivatives
    %----------------------------------------------------------------------
    if L > F(end) || iE < 3
   
        % save model-states (for each time point)
        %==================================================================
        for t = 1:length(qU)
 
            % states
            %--------------------------------------------------------------
            a     = spm_unvec(qU(t).a{1},{G.a});
            v     = spm_unvec(pU(t).v{1},{G.v});
            x     = spm_unvec(pU(t).x{1},{G.x});
            z     = spm_unvec(pU(t).z{1},{G.v});
            w     = spm_unvec(pU(t).w{1},{G.x});
            for i = 1:nl
                try
                    PU.v{i}(:,t) = spm_vec(v{i});
                    PU.z{i}(:,t) = spm_vec(z{i});
                end
                try
                    PU.x{i}(:,t) = spm_vec(x{i});
                    PU.w{i}(:,t) = spm_vec(w{i});
                end
                try
                    QU.a{i}(:,t) = spm_vec(a{i});
                end
            end

            did = sprintf( '%02dn', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
            disp_variable(fid, did, "a", a    );
            disp_variable(fid, did, "v", v    );
            disp_variable(fid, did, "x", x    );
            disp_variable(fid, did, "z", z    );
            disp_variable(fid, did, "w", w    );
            disp_variable(fid, did, "PU.v", PU.v    );
            disp_variable(fid, did, "PU.z", PU.z    );
            disp_variable(fid, did, "PU.x", PU.x    );
            disp_variable(fid, did, "PU.w", PU.w    );
            disp_variable(fid, did, "QU.a", QU.a    );
 
            % conditional modes
            %--------------------------------------------------------------
            v     = spm_unvec(qU(t).v{1},{M(1 + 1:end).v});
            x     = spm_unvec(qU(t).x{1},{M(1:end - 1).x});
            z     = spm_unvec(qE{t}(1:(ny + nv)),{M.v});
            w     = spm_unvec(qE{t}((1:nx) + (ny + nv)*n),{M.x});
            for i = 1:(nl - 1)
                if M(i).m, QU.v{i + 1}(:,t) = spm_vec(v{i}); end
                if M(i).l, QU.z{i}(:,t)     = spm_vec(z{i}); end
                if M(i).n, QU.x{i}(:,t)     = spm_vec(x{i}); end
                if M(i).n, QU.w{i}(:,t)     = spm_vec(w{i}); end
            end
            QU.v{1}(:,t)  = spm_vec(qU(t).y{1}) - spm_vec(z{1});
            QU.z{nl}(:,t) = spm_vec(z{nl});
 
            did = sprintf( '%02dp', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
            disp_variable(fid, did, "v", v    );
            disp_variable(fid, did, "x", x    );
            disp_variable(fid, did, "z", z    );
            disp_variable(fid, did, "w", w    );
            disp_variable(fid, did, "QU.v", QU.v    );
            disp_variable(fid, did, "QU.z", QU.z    );
            disp_variable(fid, did, "QU.x", QU.x    );
            disp_variable(fid, did, "QU.w", QU.w    );

            % and conditional covariances
            %--------------------------------------------------------------
            i       = (1:nx);
            QU.S{t} = qC{t}(i,i);
            PU.S{t} = pC{t}(i,i);
            i       = (1:nv) + nx*n;
            QU.C{t} = qC{t}(i,i);
            PU.C{t} = pC{t}(i,i);

            disp_variable(fid, did, "QU.S", QU.S    );
            disp_variable(fid, did, "PU.S", PU.S    );
            disp_variable(fid, did, "QU.C", QU.C    );
            disp_variable(fid, did, "PU.C", PU.C    );
        end
 
        % save conditional densities
        %------------------------------------------------------------------
        B.QU  = QU;
        B.PU  = PU;
        B.qp  = qp;
        B.qh  = qh;
 
        % decrease regularisation
        %------------------------------------------------------------------
        F(iE) = L;
        te    = min(te + 1,8);
 
        did = sprintf( '%02dr', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        disp_variable(fid, did, "F", F    );
        disp_variable(fid, did, "te", te    );
    else
 
        % otherwise, return to previous expansion point and break
        %------------------------------------------------------------------
        QU    = B.QU;
        PU    = B.PU;
        qp    = B.qp;
        qh    = B.qh;
 
        % increase regularisation
        %------------------------------------------------------------------
        F(iE) = F(end);
        te    = min(te - 1,0);

        did = sprintf( '%02ds', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
        disp_variable(fid, did, "QU", QU    );
        disp_variable(fid, did, "PU", PU    );
        disp_variable(fid, did, "qp", qp    );
        disp_variable(fid, did, "qh", qh    );
        disp_variable(fid, did, "F",  F     );
        disp_variable(fid, did, "te", te    );
        
    end
 
    % report and break if convergence
    %======================================================================
    if db
        figure(Fdem)
        spm_DEM_qU(QU)
        if np
            subplot(nl,4,4*nl)
            params_ = full(Up*qp.e);
            if DUMPLOG
                fprintf('Headbirths,Banner,ParametersMinusPrior,i,param\n' ); 
                for i_ = 1:length(params_)
                    fprintf('Headbirths,ADEM,ParametersMinusPrior,%d,%1.6f\n', i_, params_(i_) ); 
                end
                fprintf('Headbirths,End,ParametersMinusPrior,i,param\n' ); 
            end
            bar(params_) % Bar graph of the parameters
            xlabel({'parameters';'{minus prior}'})
            axis square, grid on
        end
        if length(F) > 2 % Headbirths only show this graph when there are at least 3 datapoints
            subplot(nl,4,4*nl - 1)
            F_minus_F1_ = F - F(1);
            if DUMPLOG
                fprintf('Headbirths,Banner,LogEvidence,i,F-F1\n' ); 
                for i_ = 1:length(F_minus_F1_)
                    fprintf('Headbirths,ADEM,LogEvidence,%d,%1.6f\n', i_, F_minus_F1_(i_) ); 
                end
                fprintf('Headbirths,End,LogEvidence,i,F-F1\n' ); 
            end
            plot(F_minus_F1_)
            xlabel('updates')
            title('log-evidence')
            axis square, grid on
        end
        drawnow
        
        % report (EM-Steps)
        %------------------------------------------------------------------
        str{1} = sprintf('ADEM: %i (%i)',iE,iM);
        str{2} = sprintf('F:%.4e',full(L - F(1)));
        str{3} = sprintf('p:%.2e',full(dp'*dp));
        str{4} = sprintf('h:%.2e',full(mh'*mh));
        str{5} = sprintf('(%.2e sec)',full(toc));
        
        fprintf('%-16s%-16s%-14s%-14s%-16s\n',str{:})
        str{1} = sprintf('Headbirths,ADEM,Report,%i,%i',iE,iM);
        fprintf('%s,%s,%s,%s,%s\n',str{:}) % Same info but in CSV format
    end
    
    if (norm(dp,1) < exp(-8)) && (norm(mh,1) < exp(-8)), break, end

    %try
    %dump_dem(DEM, iE, 'part1');
    %catch
    %fprintf('FAILURE,DUMP_DEM,\n');
    %end
end
 
% assemble output arguments
%==========================================================================

% conditional moments of model-parameters (rotated into original space)
%--------------------------------------------------------------------------
qP.P   = spm_unvec(Up*qp.e + spm_vec(M.pE),M.pE);
qP.C   = Up*qp.c*Up';
qP.V   = spm_unvec(diag(qP.C),M.pE);
 
% conditional moments of hyper-parameters (log-transformed)
%--------------------------------------------------------------------------
qH.h   = spm_unvec(qh.h,{{M.hE} {M.gE}});
qH.g   = qH.h{2};
qH.h   = qH.h{1};
qH.C   = qh.c;
qH.V   = spm_unvec(diag(qH.C),{{M.hE} {M.gE}});
qH.W   = qH.V{2};
qH.V   = qH.V{1};
 
% Fill in DEM with response and its causes
%--------------------------------------------------------------------------
DEM.pP.P = {G.pE};            % parameters encoding process

DEM.M  = M;                   % generative model
DEM.U  = U;                   % causes
DEM.Y  = PU.v{1};             % response

DEM.pU = PU;                  % prior moments of model-states
DEM.qU = QU;                  % conditional moments of model-states
DEM.qP = qP;                  % conditional moments of model-parameters
DEM.qH = qH;                  % conditional moments of hyper-parameters
 
DEM.F  = F;                   % [-ve] Free energy
try
    DEM.J  = J;               % [-ve] Free energy (over samples)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FINAL DUMP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

did = sprintf( '%02dz', iE ) ; % dump ID is the iteration number (with leading zeros for sorting into time order)
fprintf('Headbirths,DumpFInal,%s\n' , did);
fprintf('Headbirths,DumpStart,%s\n' , did);
if 0 % previously dump_dem(DEM, iE, 'part2');
    % for i=1:length(DEM.pP.P); % [1×1 struct]
    %     if(isempty(DEM.pP.P{i} )) % : [1×1 struct])
    %         fprintf(fid, "DEM.pP.P{%d} @%s =  [];\n", i, did);
    %     else
    %         disp_scalar(fid, did, "DEM.pP.P{i}.a", pP.P{i}.a ); % {0×0 double} % parameters encoding process
    %         disp_vec(fid, did,    "DEM.pP.P{i}.b", pP.P{i}.b ); % {0×0 double} % parameters encoding process
    %         disp_vec(fid, did,    "DEM.pP.P{i}.c", pP.P{i}.c ); % {0×0 double} % parameters encoding process
    %         disp_scalar(fid, did, "DEM.pP.P{i}.d", pP.P{i}.d ); % {0×0 double} % parameters encoding process
    %     end
    % end
    disp_array(fid, did, "DEM.Y", DEM.Y)      % [2×128 double] % response

    % DEM.pU  % : [1×1 struct]); % prior moments of model-states
    disp_array(fid, did, "DEM.pU.v{1}", PU.v{1} ); % : {[2×128 double]  [1×128 double]} % prior moments of model-states
    disp_vec(fid, did, "DEM.pU.v{2}", PU.v{2} ); % : {[2×128 double]  [1×128 double]}   % prior moments of model-states
    disp_array(fid, did, "DEM.pU.z{1}", PU.z{1} ); % : {[2×128 double]  [1×128 double]} % prior moments of model-states
    disp_vec(fid, did, "DEM.pU.z{2}", PU.z{2} ); % : {[2×128 double]  [1×128 double]}   % prior moments of model-states
    disp_array(fid, did, "DEM.pU.x{1}", PU.x{1} ); % : {[2×128 double]  [1×128 double]} % prior moments of model-states
    disp_vec(fid, did, "DEM.pU.x{2}", PU.x{2} ); % : {[2×128 double]  [1×128 double]}   % prior moments of model-states
    disp_array(fid, did, "DEM.pU.w{1}", PU.w{1} ); % : {[2×128 double]  [1×128 double]} % prior moments of model-states
    disp_vec(fid, did, "DEM.pU.w{2}", PU.w{2} ); % : {[2×128 double]  [1×128 double]}   % prior moments of model-states

    % DEM.qU
    disp_vec(fid, did, "DEM.qU.a{1}", QU.a{1} ); % : {[0×128 double]  [1×128 double]}   % conditional moments of model-states
    disp_vec(fid, did, "DEM.qU.a{2}", QU.a{2} ); % : {[0×128 double]  [1×128 double]}   % conditional moments of model-states
    disp_array(fid, did, "DEM.qU.v{1}", QU.v{1} ); % : {[2×128 double]  [1×128 double]} % conditional moments of model-states
    disp_vec(fid, did, "DEM.qU.v{2}", QU.v{2} ); % : {[2×128 double]  [1×128 double]}   % conditional moments of model-states
    disp_array(fid, did, "DEM.qU.z{1}", QU.z{1} ); % : {[2×128 double]  [1×128 double]} % conditional moments of model-states
    disp_vec(fid, did, "DEM.qU.z{2}", QU.z{2} ); % : {[2×128 double]  [1×128 double]}   % conditional moments of model-states
    disp_array(fid, did, "DEM.qU.x{1}", QU.x{1} ); % : {[2×128 double]} % conditional moments of model-states
    disp_array(fid, did, "DEM.qU.w{1}", QU.w{1} ); % : {[2×128 double]} % conditional moments of model-states
    disp_vec(fid, did, "DEM.qU.S{1}", full(DEM.qU.S{1}) ); % : {1×128 cell} % conditional moments of model-states
    disp_vec(fid, did, "DEM.qU.C{1}", full(DEM.qU.C{1}) ); % : {1×128 cell} % conditional moments of model-states

    % DEM.qP.P: {[1×1 struct]  []} % conditional moments of model-parameters
    for i=1:length(qP.P)
        prefix = sprintf('DEM.qP.P{%d}', i);
        if(isempty(qP.P{i})) % : [1×1 struct])
            fprintf(fid, "DEM.qP.P{%d} @%s =  [];\n", i, did);
        else
            disp_scalar(fid, did,strcat(prefix, ".a"), qP.P{i}.a); % a: 2.4610
            disp_vec(fid, did,strcat(prefix, ".b"),    qP.P{i}.b); % b: [-1.1237 3.1822]
            disp_vec(fid, did,strcat(prefix, ".c"),    qP.P{i}.c); % c: [-1.1995 -2.1791 -2.1791 0.6481]
            disp_scalar(fid, did,strcat(prefix, ".d"), qP.P{i}.d); % d: 0
        end
    end
    disp_array(fid, did, "DEM.qP.C", full(qP.C) ); %: [8×8 double]

    % DEM.qP.V: {[1×1 struct]  []} % conditional moments of model-parameters
    for i=1:length(qP.V)
        prefix = sprintf('DEM.qP.V{%d}', i);
        if(isempty(qP.V{i})) % : [1×1 struct])
            fprintf(fid, "DEM.qP.V{%d} @%s =  [];\n", i, did);
        else
            disp_scalar(fid, did,strcat(prefix, ".a"), qP.V{i}.a); % a: 2.4610
            disp_vec(fid, did,strcat(prefix, ".b"),    qP.V{i}.b); % b: [-1.1237 3.1822]
            disp_vec(fid, did,strcat(prefix, ".c"),    qP.V{i}.c); % c: [-1.1995 -2.1791 -2.1791 0.6481]
            disp_scalar(fid, did,strcat(prefix, ".d"), qP.V{i}.d); % d: 0
        end
    end

    disp_vec(fid, did, "DEM.qH.h{1}", full(qH.h{1})); % : {[0×1 double]  [0×1 double]}
    disp_vec(fid, did, "DEM.qH.h{2}", full(qH.h{2})); % : {[0×1 double]  [0×1 double]}
    disp_vec(fid, did, "DEM.qH.g{1}", full(qH.g{1})); % : {[0×1 double]  [0×1 double]}
    disp_vec(fid, did, "DEM.qH.g{2}", full(qH.g{2})); % : {[0×1 double]  [0×1 double]}
    if isempty(qH.C) % : []
        fprintf(fid, "DEM.qH.C @%s =  [];\n", i, did);
    else
        disp_array(fid, did, "DEM.qH.C", full(qH.C) ); %: [0x0 double]
    end
    disp_vec(fid, did, "DEM.qH.V{1}", full(qH.V{1})); % : {[0×1 double]  [0×1 double]} % conditional moments of hyper-parameters
    disp_vec(fid, did, "DEM.qH.V{2}", full(qH.V{2})); % : {[0×1 double]  [0×1 double]} % conditional moments of hyper-parameters
    disp_vec(fid, did, "DEM.qH.W{1}", full(qH.W{1})); % : {[0×1 double]  [0×1 double]} % conditional moments of hyper-parameters
    disp_vec(fid, did, "DEM.qH.W{2}", full(qH.W{2})); % : {[0×1 double]  [0×1 double]} % conditional moments of hyper-parameters
    disp_scalar(fid, did, "DEM.F", F)      % 1.9967e+03 % [-ve] Free energy
    try
        disp_vec(fid, did, "DEM.J", J)      % [1×128 double] % [-ve] Free energy (over samples)
    end
    %DEM.U  = U; All zero sparse: 1×128                   ) 
end
fprintf(fid, "<END> @%s\n", did);
fclose(fid_to_close);
fprintf("END,DUMP_DEM,%d,\n");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END FINAL DUMP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Headbirths,Footer,ADEM,Dynamic Expectation Maximization algorithm with Active inversion\n' );

