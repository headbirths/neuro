
function disp_vec(fid, did, signame, v)
    if isempty(v)
        fprintf(fid, "%s @%s =  [] ;\n", signame, did);
    else
        for i=1:length(v)
            %fprintf(fid, "%s( %d ) = %.6e ; %%$ %s\n", signame, i, v(i), did);
            fprintf(fid, "%s(%04d) @%s =  %.6e ;\n", signame, i, did, v(i)); % @ notation for sorting entries in time-order
        end
    end
end