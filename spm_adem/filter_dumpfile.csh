########################################################
# Filters the ADEM_Learning dump file down to something
# read repeatedly into the Python script.
# Also for sifting through dumpfile contents
########################################################

# Input: sort; remove timestep duplications
sort  dump_dem_202006262347.txt > sorted_202006262347.txt
# filter out duplication of values...
cat sorted_202006262347.txt | awk 'BEGIN{s=l=v=""}{if(($1!=s)||($NF!=v)){print l; s=$1; v=$NF; l=$0}}END{print l}' > changes_202006262347.txt
# Filter out intermediate values: one value per timestep/stage
cat changes_202006262347.txt | awk 'BEGIN{s=t=l=""}{if(NF>2){if(($1!=s)||($(NF-2)!=t)){print l; s=$1; t=$(NF-2); l=$0}}}END{print l}' > strobed_202006262347.txt

# The following is just successive filtering to help find information worth plotting...
#% Filter out zeroes and empties
#% 1019  cat changes_202006262347.txt | awk '{if(($NF!="0")&&($NF!="[]")){print}}' > nozero_202006262347.txt
#% 1031  cat nozero_202006262347.txt | grep -v 'dE.dp\|dE.du\|df.dx\|dFdp\|dFdu\|Dfdx' | more
#% 
#%  cat less2E_202006262347.txt | grep -v 'EiSE' | more
#%  cat less2E_202006262347.txt | grep -v 'EiSE\|Hqu.c' | more
#%  cat less2E_202006262347.txt | grep -v 'EiSE\|Hqu.c\|iY  ' | more
#%  cat less2E_202006262347.txt | grep -v 'EiSE\|Hqu.c\|iY ' | more
#%  cat less2D_202006262347.txt | awk '{s=substr($1,1,2);if((s!="E(")&&(s!="p{")){print}}' > less2E_202006262347.txt
#%  #cat less2D_202006262347.txt | awk '{s=substr($1,1,2);if((s!="E(")&&(s!="p{")&&(s!="q{")){print}}' > less2E_202006262347.txt
#%  cat less2E_202006262347.txt | grep -v 'EiSE\|Hqu.c\|iY \|pU(iY) .c{\|pU(iY) .v{\|pC{iY} (\|pU(iY) .w{\|pU(iY) .x{\|pU(iY) .z{\|pu.c\|PU.C\|PU.S\|pu.v\|PU.v\|pu.w\|PU.w\|pu.x\|PU.x\|pu.z\|PU.z' > less2P_202006262347.txt
#%  cat less2P_202006262347.txt | grep -v 'qC{\|qE{\|qp.c (\|qp.e(\|qp.ic(\|qU(iY) .c\|qU(iY) .u\|qU(iY) .v\|qU(iY) .x\|qU(iY) .y\|qu.c (\|qu.c{\|QU.C{\|QU.S{\|qu.u\|qu.v\|QU.v\|QU.w\|qu.x\|QU.x\|qu.y\|QU.z' > less2Q_202006262347.txt
#%  cat less2Q_202006262347.txt | more
#%  1105  history
#% 
#% 
