function im= imScale(input_im,scale)
% caution: disable antialiasing
    img_height=size(input_im,1);
    img_width =size(input_im,2);

    
    if img_height>img_width
        
        img_width=scale;
        img_height=NaN;
    else
        
        img_width=NaN;
        img_height=scale;
    end
    
    im=imresize(input_im,[img_height,img_width],'bilinear','antialiasing',false);

end