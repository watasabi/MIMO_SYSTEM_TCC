function [GGRv] = GGR(obj_no,GreyWolves_num,GreyWolves,Cost_GreyWolves)
    GMR = zeros(1,GreyWolves_num);
    GD = zeros(1,GreyWolves_num);
    for ii = 1:GreyWolves_num
        soma1 = 0;
        soma2 = 0;
        for j = 1:GreyWolves_num
            if ii ~= j
              if obj_no == 2 
               dif = Cost_GreyWolves(ii,1)*Cost_GreyWolves(ii,2) - Cost_GreyWolves(j,1)*Cost_GreyWolves(j,2); 
              end  
              if obj_no == 3
              dif = Cost_GreyWolves(ii,1)*Cost_GreyWolves(ii,2)*Cost_GreyWolves(ii,3) - Cost_GreyWolves(j,1)*Cost_GreyWolves(j,2)*Cost_GreyWolves(ii,3); 
              end
              soma1 = soma1 +  max(dif,0);
               d = norm(GreyWolves(ii,:) - GreyWolves(j,:));
              soma2 = soma2 + d;
            end
        end
    GMR(1,ii) = soma1;
    GD(1,ii) = soma2;
    end
    GGRv = GMR./GD;     

end

