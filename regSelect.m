function groups=regSelect(bboxes,alpha)

areas=( bboxes(:,3)-bboxes(:,1) ).*(bboxes(:,4)-bboxes(:,2));
[~,idxs]=sort(areas,'descend');
bboxes=bboxes(idxs,:);




count=1;
groups(count).ubbox=bboxes(1,:);
groups(count).bboxes=bboxes(1,:);
bboxes(1,:)=[];


while ~isempty(bboxes)
    
    pre_b_size=size(bboxes,1);
    for i=1:1:count
        ratios=getMaxRatio(groups(i).ubbox ,bboxes);
        [~,r_idx]=sort(ratios,'descend');
        
        top=r_idx(1);
        if ratios(top)>alpha
            select_box=(bboxes(top,:));
            ubbox=getBoxUnion(groups(i).ubbox,select_box);
            
            pre_ratio=getMaxRatio(ubbox,groups(i).bboxes);
            
            if length(pre_ratio)== sum(pre_ratio>=alpha)
                
                bboxes(top, :)=[];
                
                if isempty(bboxes)
                   break; 
                    
                end
                groups(i).ubbox=ubbox;
                groups(i).bboxes=[groups(i).bboxes;select_box];
            end
        end
        
        
    end
    aft_b_size=size(bboxes,1);
    
    if (pre_b_size==aft_b_size ) && ~isempty(bboxes)
    
    count=count+1;
    groups(count).ubbox=bboxes(1,:);
    groups(count).bboxes=bboxes(1,:);
    bboxes(1,:)=[];
    end
    
    
    
    
    
end





%algin the bounding boxes inside each reigon to the region
for i=1:1:count
    groups(i).bboxes=groups(i).bboxes- repmat(  [groups(i).ubbox(1),groups(i).ubbox(2),groups(i).ubbox(1),groups(i).ubbox(2)], size(groups(i).bboxes,1),1)+1;
end



end


function   ratios=getMaxRatio(initial_region,bboxes)
unionboxes=getBoxUnion(initial_region,bboxes);

areas=(bboxes(:,3)-bboxes(:,1)).*(bboxes(:,4)-bboxes(:,2));
unionAreas=(unionboxes(:,3)-unionboxes(:,1)).*(unionboxes(:,4)-unionboxes(:,2));

ratios=areas./unionAreas;



end


function box = getBoxUnion(rect, rects)

xrange = getSegUnion([rect(1), rect(3)], [rects(1), rects(3)]);
yrange = getSegUnion([rect(2), rect(4)], [rects(2), rects(4)]);
box = [xrange(:,1), yrange(:,1), xrange(:,2) , yrange(:,2) ];
end


function unionSeg = getSegUnion(ev1, ev2)
unionSeg = [min(ev1(1), ev2(:,1)), max(ev1(2), ev2(:,2))];
end