//
// 2026 Helios CV enter examination
//
/*
 * ██   ██ ███████ ██      ██  ██████  ███████
 * ██   ██ ██      ██      ██ ██    ██ ██
 * ███████ █████   ██      ██ ██    ██ ███████
 * ██   ██ ██      ██      ██ ██    ██      ██
 * ██   ██ ███████ ███████ ██  ██████  ███████
 */
#ifndef GRAPHIC_DISPLAY_BOARD_H
#define GRAPHIC_DISPLAY_BOARD_H

class DisplayBoard
{
private:
    float object_pos, target_pos, current_pos;

public:
    DisplayBoard();

    void update(const float &object_pos, const float &target_pos, const float &current_pos);
    void show();
};

#endif // GRAPHIC_DISPLAY_BOARD_H