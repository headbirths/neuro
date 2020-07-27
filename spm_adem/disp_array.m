
function disp_array(fid, did, signame, v)
    if isempty(v)
        fprintf(fid, "%s @%s =  [] ;\n", signame, did);
    elseif iscell(v)
        if isempty(v)
            fprintf(fid, "%s @%s =  [] ;\n", signame, did);
        else
            size(v, 1)
            for i=1:size(v, 1)
                i
                if isempty(vc{i})
                    fprintf(fid, "%s{%d} @%s =  [] ;\n", signame, i, did);
                else
                    vcc = vc{i};
                    length(vcc)
                    for j=1:length(vcc)
                        j
                        %fprintf(fid, "%s( %d , %d ) = %.6e ; %% HELLO \n", signame, i, j, v{i}(j));
                        %fprintf(fid, "%s( %d , %d ) = %.6e ; %%$ %s \n", signame, i, j, v(i,j), did);
                        fprintf(fid, "%s{%04d}(%04d) @%s =  %.6e ;\n", signame, i, j, did, full(vc(j))); % @ notation for sorting entries in time-order
                        %fprintf(fid, "HELLO\n");
                    end
                end
            end
        end
    else
        for i=1:size(v, 1)
            for j=1:size(v,2)
                %fprintf(fid, "%s( %d , %d ) = %.6e ; %% HELLO \n", signame, i, j, v(i,j));
                %fprintf(fid, "%s( %d , %d ) = %.6e ; %%$ %s \n", signame, i, j, v(i,j), did);
                fprintf(fid, "%s(%04d,%04d) @%s =  %.6e ;\n", signame, i, j, did, v(i,j)); % @ notation for sorting entries in time-order
                %fprintf(fid, "HELLO\n");
            end
        end
    end
end




% function disp_array(fid, did, signame, v)
%     if isempty(v)
%         fprintf(fid, "%s @%s =  [] ;\n", signame, did);
%     % elseif iscell(v)
%     %     for i=1:size(v, 1)
%     %         vc = full(v{i});
%     %         for j=1:length(vc)
%     %             %fprintf(fid, "%s( %d , %d ) = %.6e ; %% HELLO \n", signame, i, j, v{i}(j));
%     %             %fprintf(fid, "%s( %d , %d ) = %.6e ; %%$ %s \n", signame, i, j, v(i,j), did);
%     %             fprintf(fid, "%s{%04d}(%04d) @%s =  %.6e ;\n", signame, i, j, did, full(vc(j))); % @ notation for sorting entries in time-order
%     %             %fprintf(fid, "HELLO\n");
%     %         end
%     %     end
%     else
%         for i=1:size(v, 1)
%             for j=1:size(v,2)
%                 %fprintf(fid, "%s( %d , %d ) = %.6e ; %% HELLO \n", signame, i, j, v(i,j));
%                 %fprintf(fid, "%s( %d , %d ) = %.6e ; %%$ %s \n", signame, i, j, v(i,j), did);
%                 fprintf(fid, "%s(%04d,%04d) @%s =  %.6e ;\n", signame, i, j, did, v(i,j)); % @ notation for sorting entries in time-order
%                 %fprintf(fid, "HELLO\n");
%             end
%         end
%     end
% end