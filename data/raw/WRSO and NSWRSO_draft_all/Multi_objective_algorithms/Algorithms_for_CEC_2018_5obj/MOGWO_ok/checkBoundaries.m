function [POS] = checkBoundaries(POS,var_max,var_min)
    % Useful matrices
    Np = size(POS,1);
    MAXLIM   = repmat(var_max(:)',Np,1);
    MINLIM   = repmat(var_min(:)',Np,1);

    % Correct positions and velocities
    POS(POS>MAXLIM) = MAXLIM(POS>MAXLIM);
    POS(POS<MINLIM) = MINLIM(POS<MINLIM);
end