
function disp_scalar(fid, did, signame, v)
    if isempty(v)
        fprintf(fid, "%s @%s =  [] ;\n", signame, did);
    else
        %fprintf(fid, "%s = %.6e ; %%$ %s\n", signame, v, id);
        fprintf(fid, "%s @%s =  %.6e ;\n", signame, did, v); % @ notation for sorting entries in time-order
    end
end