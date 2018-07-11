% generate initial subpixel mapping using fraction image
% by Lavender
function [ini_sub_map,mat_n_class]=generate_ini_submap(LSU_Original_sub,S)
% calculate number of subpixels for each class
mat_n_class=round(LSU_Original_sub*S*S);
for i=1:size(LSU_Original_sub,1)
    for j=1:size(LSU_Original_sub,2)
        if sum(mat_n_class(i,j,:))~=S*S
            [sort_numbers,sort_index]=sort(mat_n_class(i,j,:),'descend');
            mat_n_class(i,j,sort_index(1))=mat_n_class(i,j,sort_index(1))+S*S-sum(mat_n_class(i,j,:));
        end
    end
end

% randomly distribute subpixels according to class number
ini_sub_map=zeros(size(LSU_Original_sub,1)*S,size(LSU_Original_sub,2)*S);
n_temp=0;
for i=1:size(LSU_Original_sub,1)
    for j=1:size(LSU_Original_sub,2)
        % generate random unique index inside each coarse pixel
        rand_index=randperm(S*S);
        % index to subscripts inside each coarse pixel
        [sub_row,sub_column] = ind2sub([S,S],rand_index);
        class_temp=[];
        for k=1:size(mat_n_class,3)
            n_temp=mat_n_class(i,j,k);
            class_temp=[class_temp ones(1,n_temp)*k];
        end
        ini_sub_map((i-1)*S+1:i*S,(j-1)*S+1:j*S)=reshape(class_temp(rand_index),S,S);
    end
end
