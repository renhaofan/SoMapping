#include <QApplication>
#include <QDebug>
int main(int argc, char *argv[]) {
  // Initialize application
  QApplication a(argc, argv);

  qDebug() << "Hello world";
  // Start event loop
  return a.exec();
}
