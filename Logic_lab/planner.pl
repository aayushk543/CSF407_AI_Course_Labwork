% =============================================================================
% Laboratory - Logical Reasoning for Planning
% Tasks 6, 7, and 8: Prolog Plan Verifier Knowledge Base
% =============================================================================

% Task 6: Connected warehouse locations
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Rule: The robot can move between connected locations
can_move(X, Y) :-
    connected(X, Y).

% Task 7: Valid move checker for proposed plans
valid_move(X, Y) :-
    connected(X, Y).

% Task 8: Rule-based inference chain
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% =============================================================================
% Example Queries for Verification:
% ?- can_move(a, b).      % Expected: true.
% ?- can_move(a, c).      % Expected: false. (no direct link between a and c)
% ?- valid_move(a, b).    % Expected: true.
% ?- valid_move(b, c).    % Expected: true.
% ?- valid_move(a, c).    % Expected: false. (detects invalid jump)
% ?- reduce_speed.        % Expected: true. (via wet_road => slippery => reduce_speed)
% =============================================================================
