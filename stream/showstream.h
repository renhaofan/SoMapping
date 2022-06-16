#ifndef SHOWSTREAM_H
#define SHOWSTREAM_H

#include "grabstream.h"
#include <QWidget>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QPushButton>
#include <QVBoxLayout>

// to solve cpu working problem when pause
#include <atomic>
#include <QMutex>
#include <QWaitCondition>

class ShowStream : public QWidget
{
    Q_OBJECT

public:
    ShowStream(QWidget *parent = nullptr);
    ~ShowStream();

private slots:
    void slotGetFrame(QImage color, QImage depth);
    void slotStartBtnclicked();
    void slotPauseBtnclicked();
    void slotResumeBtnclicked();
    void slotStopBtnclicked();
    void slotClearBtnclicked();


private:
    QPushButton *m_startBtn;
    QPushButton *m_pauseBtn;
    QPushButton *m_resumeBtn;
    QPushButton *m_stopBtn;
    QPushButton *m_clearBtn;



    GrabStream *m_thread;

    QGraphicsScene *m_sceneColor;
    QGraphicsScene *m_sceneDepth;

    QGraphicsView *m_viewColor;
    QGraphicsView *m_viewDepth;


    QImage m_sideColor;
    QImage m_sideDepth;

    QVBoxLayout *vlayout;

    QMutex mutex;
    QWaitCondition condition;

    void setShowWidgetDefault();
};




#endif // SHOWSTREAM_H
