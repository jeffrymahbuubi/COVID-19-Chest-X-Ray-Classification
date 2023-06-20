function data = editimagever2(data)% added lines: 
data = imresize(data,[224 224]);
data = data(:,:,min(1:3, end)); 

data = cat(3, data);
end