#include "showstream.h"
#include "pcolor.h"
#include <QDebug>


ShowStream::ShowStream(QWidget *parent)
    : QWidget(parent)
{
    m_thread = new GrabStream;
    connect(m_thread, &GrabStream::sigSendFrame, this, &ShowStream::slotGetFrame);

    m_startBtn = new QPushButton;
    m_startBtn->setText("Start");
    m_pauseBtn = new QPushButton;
    m_pauseBtn->setText("Pause");
    m_resumeBtn = new QPushButton;
    m_resumeBtn->setText("Resume");
    m_stopBtn = new QPushButton;
    m_stopBtn->setText("Stop");
    m_clearBtn = new QPushButton;
    m_clearBtn->setText("clear");

    connect(m_startBtn, &QPushButton::clicked, this, &ShowStream::slotStartBtnclicked);
    connect(m_pauseBtn, &QPushButton::clicked, this, &ShowStream::slotPauseBtnclicked);
    connect(m_resumeBtn, &QPushButton::clicked, this, &ShowStream::slotResumeBtnclicked);
    connect(m_stopBtn, &QPushButton::clicked, this, &ShowStream::slotStopBtnclicked);
    connect(m_clearBtn, &QPushButton::clicked, this, &ShowStream::slotClearBtnclicked);

    m_sceneColor = new QGraphicsScene;
    m_sceneColor->addPixmap(QPixmap(":/default/images/no_image320240.png"));
    m_viewColor = new QGraphicsView(m_sceneColor);
//    m_viewColor->setStyleSheet("padding: 0px; border: 0px;");

    m_sceneDepth = new QGraphicsScene;
    m_sceneDepth->addPixmap(QPixmap(":/default/images/no_image320240.png"));
    m_viewDepth = new QGraphicsView(m_sceneDepth);
//    m_viewDepth->setStyleSheet("padding: 0px; border: 0px;");



    vlayout = new QVBoxLayout;
    vlayout->addWidget(m_startBtn);
    vlayout->addWidget(m_pauseBtn);
    vlayout->addWidget(m_resumeBtn);
    vlayout->addWidget(m_stopBtn);
    vlayout->addWidget(m_clearBtn);

    vlayout->addWidget(m_viewColor);
    vlayout->addWidget(m_viewDepth);


    this->setLayout(vlayout);

}

ShowStream::~ShowStream()
{
//    m_thread->setStop();
}

void ShowStream::setShowWidgetDefault()
{
    m_sceneColor->clear();
    m_sceneColor->addPixmap(QPixmap(":/default/images/no_image320240.png"));
    m_sceneColor->update();

    m_sceneDepth->clear();
    m_sceneDepth->addPixmap(QPixmap(":/default/images/no_image320240.png"));
    m_sceneDepth->update();

}

void ShowStream::slotStartBtnclicked()
{
    qDebug() << "start";
    if (!m_thread->isRunning())
    {
        m_thread->setRunning();
        m_thread->start();
    }
}

void ShowStream::slotPauseBtnclicked()
{
    m_thread->setPause();
}

void ShowStream::slotResumeBtnclicked()
{
    m_thread->setResume();
}

void ShowStream::slotClearBtnclicked()
{
    if (!m_thread->isRunning())
    {
        setShowWidgetDefault();
    }
}



void ShowStream::slotStopBtnclicked()
{
    // clear memory results in bugs
    // ASSERT failure in QMutexLocker: "QMutex pointer is misaligned",
    m_thread->setStop();
//    setShowWidgetDefault();
    qDebug() << "quit slotStopBtnclicked";
}

void ShowStream::slotGetFrame(QImage color, QImage depth)
{
    m_sideDepth = depth.copy();
    m_sideDepth = convertGray16ToGray8(m_sideDepth);
    m_sideDepth = image2turbo(m_sideDepth);
    m_sideDepth = m_sideDepth.scaled(0.5 * m_sideDepth.size(), Qt::KeepAspectRatio);
    m_sceneDepth->clear();
    m_sceneDepth->addPixmap(QPixmap::fromImage(m_sideDepth));
    m_sceneDepth->update();


    m_sideColor = color.copy();
    m_sideColor = m_sideColor.scaled(m_sideDepth.size());
    m_sceneColor->clear();
    m_sceneColor->addPixmap(QPixmap::fromImage(m_sideColor));
    m_sceneColor->update();
}
