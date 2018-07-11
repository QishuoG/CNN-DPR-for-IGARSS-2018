% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
% im = reshape(im, 9, 9, 4, []) ;
labels = imdb.images.label(1,batch) ;