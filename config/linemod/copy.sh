declare -a arr=("glue" "ape" "cat" "phone" "eggbox" "benchvise" "lamp" "camera" "can" "driller" "duck" "holepuncher" "iron"  )

#create training scripts
for seq in "${arr[@]}"
do
   echo "$seq"
   cat template_fw0.5.yml > "$seq"_fw0.5.yml
   sed -i "s/SEQ_NAME/$seq/g" "$seq"_fw0.5.yml
done

arraylength=${#arr[@]}
