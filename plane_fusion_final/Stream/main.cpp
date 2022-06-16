#include "mainwindow.h"
#include "showstream.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();


    ShowStream ss;
    ss.show();
    return a.exec();
}
