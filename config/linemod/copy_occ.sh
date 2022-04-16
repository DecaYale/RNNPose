declare -a arr=("glue" "ape" "cat" "phone" "eggbox" "benchvise" "lamp" "camera" "can" "driller" "duck" "holepuncher" "iron"  )

#create training scripts
for seq in "${arr[@]}"
do
   echo "$seq"
   cat template_fw0.5_occ.yml > "$seq"_fw0.5_occ.yml
   sed -i "s/SEQ_NAME/$seq/g" "$seq"_fw0.5_occ.yml
done


