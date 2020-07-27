
function disp_variable(fid, did, signame, v)
    %fprintf(fid, "<DEBUG %s>\n", signame);
    if isempty(v)
        fprintf(fid, "%s @%s e= []\n", signame, did);
    elseif iscell(v)
        %fprintf(fid, "%% %s <CELL>\n", signame);
        for i = 1:length(v)
            signame_i = sprintf("%s{%d}", signame, i);
            disp_variable(fid, did, signame_i, v{i});
        end
    elseif isstruct(v)
        %fprintf(fid, "%% %s <STRUCT>\n", signame);
        v_cell = struct2cell(v);
        v_fieldnames = fieldnames(v);
        for i = 1:length(v_cell)
            signame_i = sprintf("%s.%s", signame, v_fieldnames{i});
            disp_variable(fid, did, signame_i, v_cell(i));
        end
    elseif issparse(v)
        %fprintf(fid, "%% %s <SPARSE>\n", signame);
        signame_i = sprintf("%s$", signame); %  $  represents 'sparse'
        disp_variable(fid, did, signame, full(v));
    elseif ~isscalar(v) % i.e. is a vector
        %fprintf(fid, "%% %s <VECTOR>\n", signame);
        for i = 1:length(v)
            if iscell(v)
                signame_i = sprintf("%s{%d}", signame, i);
                disp_variable(fid, did, signame_i, full(v{i}));
            else
                signame_i = sprintf("%s(%d)", signame, i);
                disp_variable(fid, did, signame_i, full(v(i)));
            end
        end
    elseif isa(v, 'string') || isa(v, 'char')
        fprintf(fid, "%s @%s s= '%s'\n", signame, did, v); % @ notation for sorting entries in time-order
    elseif isa(v, 'float')
        rounded = round(v); % Short integer form to reduce files size
        if (abs(v) <= 999999) && (rounded == v) % small-ish integer
            fprintf(fid, "%s @%s r= %d\n", signame, did, rounded);
        else
            fprintf(fid, "%s @%s f= %.6e\n", signame, did, v);
        end
    elseif isa(v, 'integer') || isa(v, 'logical')
        fprintf(fid, "%s @%s i= %d\n", signame, did, v);
    elseif isa(v, 'function_handle')
        fprintf(fid, "%s @%s f= ?\n", signame, did);
    else
        fprintf(fid, "%s @%s o= ?\n", signame, did);
    end
end

