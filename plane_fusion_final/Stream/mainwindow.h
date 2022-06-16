#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "showstream.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    ShowStream *stream;
};
#endif // MAINWINDOW_H
