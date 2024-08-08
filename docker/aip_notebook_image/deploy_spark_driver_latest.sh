#!/usr/bin/env bash
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH

Green_font_prefix="\033[32m"
Red_font_prefix="\033[31m"
Yellow_font_prefix="\033[33m"
Font_color_suffix="\033[0m"
Info="${Green_font_prefix}[INFO]${Font_color_suffix}"
Error="${Red_font_prefix}[ERROR]${Font_color_suffix}"
Tip="${Green_font_prefix}[TIP]${Font_color_suffix}"
Notice="${Yellow_font_prefix}[NOTICE]${Font_color_suffix}"

#--------------------------- 1. Set ENV ----------------------------
function checkRoot() {
    # check if the user is root
    if [ "$(id -u)" -ne 0 ]; then
        echo -e " ${Error} Please run me as root"
        exit 2
    fi
}

function mkTmp() {
    # tmp dir for some packages
    tmp_dir="/tmp/install_driver_client"
    if [ -d "$tmp_dir" ]; then
        rm -rf $tmp_dir
    fi
    mkdir -p $tmp_dir
    mkdir -p /usr/share/java/
    mkdir -p /hadoop/spark/sparklocaldir/ && chmod 777 /hadoop/spark/sparklocaldir/
}

function setVariable() {
    FILE_SERVER="https://di-nexus-repo.idata.shopeemobile.com/repository/file_server"
    ENV_FILE="/etc/profile.d/driver.sh"

    # hadoop-client
    hadoop_client_bin="hadoop-2.10.sdi-060"
    hadoop_client_conf="hadoop-client-Prod-TL0-sdi-135"

    # hadoop 3.2.0
    hadoop_3_2_bin="hadoop-3.2.0-sdi-001"

    # hadoop external lib
    hadoop_external_lib="hadoop-java-puliclib-003"

    # spark-2.4
    spark2_bin="spark-2.4.7-sdi-035-bin-2.10.sdi-008"

    # spark-3.1
    spark3_bin="spark-3.1.2-sdi-037-bin-3.2.sdi-014"

    # spark-3.2
    spark3_2_bin="spark-3.2.1-sdi-019-bin-3.3.sdi-036"

    # spark external lib
    spark_external_lib="spark_external"

    # hbase bin and conf
    hbase_bin="hbase-1.4.12"
    hbase_config="hbase_config.tar.bz2"
    hbase_shell_tools="shells.tar.bz2"

    # phoenix
    phoenix_bin="phoenix-4.14.0-HBase-1.4-shopee-190424"
    phoenix_config="phoenix_config"
    phoenix_config_link="phoenix-4.14"

    # clickhouse
    clickhouse_bin="clickhouse-client-SClickhouse-21.8.4.1-010"

    # livy command tool
    livy_command_tool="livy-command-tool-0.8.0-incubating-sdi-020-bin"

    # node_exporter
    node_exporter="node_exporter-0.17.0.linux-amd64"

    # miniconda2
    miniconda2="miniconda2.tar"

    # version
    version=v1.0.8
    version_file=/opt/di-driver-update/version
}

#------------------------- 2. Check OS, JDK Begin ------------------------
function check_sys() {
    if [[ -f /etc/redhat-release ]]; then
        release="centos"
    elif cat /etc/issue | grep -q -E -i "debian"; then
        release="debian"
    elif cat /etc/issue | grep -q -E -i "ubuntu"; then
        release="ubuntu"
    elif cat /etc/issue | grep -q -E -i "centos|red hat|redhat"; then
        release="centos"
    elif cat /proc/version | grep -q -E -i "debian"; then
        release="debian"
    elif cat /proc/version | grep -q -E -i "ubuntu"; then
        release="ubuntu"
    elif cat /proc/version | grep -q -E -i "centos|red hat|redhat"; then
        release="centos"
    fi
    #bit=`uname -m`
}

function check_os() {
    if [ $(cat /etc/os-release | egrep -i ubuntu | wc -l) -ge 1 ]; then
        echo -e "ubuntu"
    elif [ $(cat /etc/os-release | egrep -i centos | wc -l) -ge 1 ]; then
        echo -e "centos"
    fi
}

function check_jdk() {
    echo -e "${Info} ==> Checking JDK, better install it by yourself ..."
    if [ "$(check_os)" == "ubuntu" ]; then
        apt-get install -y bzip2 >/dev/null 2>&1
        java -version >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${Info} ==> Already installed jdk"
        else
            dpkg -l | egrep jdk
            if [ $? -eq 0 ]; then
                echo -e "${Info} ==> Already installed jdk"
            else
                ls /usr/lib/jvm/ 2>/dev/null | egrep java | egrep jdk
                if [ $? -eq 0 ]; then
                    echo -e "${Info} ==> Already installed Java"
                else
                    echo -e "${Info} ==> Need to install java [wait a minute]"
                    sleep 3
                    apt-get update >/dev/null && apt-get install -y openjdk-8-jre bzip2 >/dev/null
                fi
            fi
        fi
    elif [ "$(check_os)" == "centos" ]; then
        yum install -y bzip2 >/dev/null 2>&1
        java -version >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${Info} ==> Already installed jdk"
        else
            rpm -qa | egrep jdk
            if [ $? -eq 0 ]; then
                echo -e "${Info} ==> Already installed jdk"
            else
                ls /usr/lib/jvm/ 2>/dev/null | egrep java | egrep jdk
                if [ $? -eq 0 ]; then
                    echo -e "${Info} ==> Already installed Java"
                else
                    echo -e "${Info} Need to install java [wait a minute]"
                    sleep 3
                    yum update >/dev/null && yum install -y java-1.8.0-openjdk bzip2 >/dev/null
                fi
            fi
        fi
    else
        echo -e "${Error} Not supported OS"
        exit 2
    fi
    if [ ! -L /etc/alternatives/jre ]; then
        jre_source=$(find /usr/lib/jvm -name "jre" -type d | head -1)
        ln -s $jre_source /etc/alternatives/
    fi
}

function removeFolder() {
    local choice=$1
    if [ -d "$choice" ]; then
        if [ ! -L "$choice" ]; then
            echo -e "${Notice} $choice is not a link, will mv it to ${choice}_bak"
            mv "$choice" "${choice}"_bak
        fi
    fi
}

function add_to_motd() {
    if [ -f /etc/motd ]; then
        N=$(cat /etc/motd | egrep "Shopee Data Infra" | wc -l)
        if [ $N -gt 0 ]; then
            echo -e "${Info} /etc/motd already added shopee data infra"
        else
            cat >>/etc/motd <<EOF

===========================================
============ Shopee Data Infra ============
===========================================

* Hadoop Home:          /usr/share/hadoop-client
* Spark Default:        /usr/share/spark     [link to spark-2.4]
* Hbase Default:        /usr/share/hbase
* Phoenix Default:      /usr/share/phoenix
* clickhouse Default:   /usr/share/clickhouse
* Spark 2.4.x:          /usr/share/spark-2.4
* Spark 3.1.x:          /usr/share/spark-3.1
* Spark 3.2.x:          /usr/share/spark-3.2
* Livy command tool:    /usr/share/livy
* Hbase 1.4.12:         /opt/hbase-1.4.12
* Phoenix 4.14.0:       /opt/phoenix-4.14.0-HBase-1.4-shopee-190424
* Documentation:        https://confluence.shopee.io/display/SDPS/Spark
* Support:              https://jira.shopee.io/servicedesk/customer/portal/20
    
EOF
        fi
    else
        cat >>/etc/motd <<EOF

===========================================
============ Shopee Data Infra ============
===========================================

* Hadoop Home:          /usr/share/hadoop-client
* Spark Default:        /usr/share/spark     [link to spark-2.4]
* Hbase Default:        /usr/share/hbase
* Phoenix Default:      /usr/share/phoenix
* clickhouse Default:   /usr/share/clickhouse
* Spark 2.4.x:          /usr/share/spark-2.4
* Spark 3.1.x:          /usr/share/spark-3.1
* Spark 3.2.x:          /usr/share/spark-3.2
* Livy command tool:    /usr/share/livy
* Hbase 1.4.12:         /opt/hbase-1.4.12
* Phoenix 4.14.0:       /opt/phoenix-4.14.0-HBase-1.4-shopee-190424
* Documentation:        https://confluence.shopee.io/display/SDPS/Spark
* Support:              https://jira.shopee.io/servicedesk/customer/portal/20

EOF
    fi
}

function setup_env() {
    if [ -d "/opt/cloudera" ]; then
        ENV_FILE="/etc/driver.sh"
    fi
    echo -e "${Info} Setup the all component environment variable."
    echo -e '# All env variables' >$ENV_FILE
    {
        echo -e '# Hadoop env variables'
        echo -e 'PATH=$PATH:$HOME/bin'
        echo -e 'export HADOOP_HOME=/usr/share/hadoop-client'
        echo -e 'export HADOOP_CONF_DIR=/etc/hadoop-client'
        echo -e 'export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/include:$HADOOP_HOME/lib:$HADOOP_HOME/libexec:$HADOOP_HOME/sbin:$HADOOP_HOME/share:$PATH'
        echo ''

        echo -e '# Java env variables'
        echo -e 'export JAVA_HOME=/etc/alternatives/jre'
        echo ''

        echo -e '# Spark env variables'
        echo -e 'export PATH=$PATH:/usr/share/spark/bin:/usr/share/spark2/bin:/usr/share/spark3/bin'
        echo -e 'export PATH=$PATH:/usr/share/miniconda2/envs/py27/bin'
        echo ''

        echo -e '# Hbase env variables'
        echo -e 'export HBASE_HOME=/usr/share/hbase'
        echo -e 'export HBASE_CONF=/etc/hbase'
        echo -e 'export PATH=/usr/share/hbase/bin:$PATH'
        echo -e 'export PATH=/usr/share/hbase_tools/shells/hbase_shell:/usr/share/hbase_tools/shells/phoenix_shell/:$PATH'
        echo ''

        echo -e '# Phoenix env variables'
        echo -e 'export HBASE_CONF_DIR=/etc/phoenix'
        echo -e 'export PHOENIX_HOME=/usr/share/phoenix'
        echo ''

        echo -e "# Miniconda env variables"
        echo -e 'export MINICONDA2_HOME=/usr/share/miniconda2'
        echo -e 'export PATH=$MINICONDA2_HOME/bin:$MINICONDA2_HOME/envs/py27/bin:$MINICONDA2_HOME/envs/py27/include:$MINICONDA2_HOME/envs/py27/lib:$MINICONDA2_HOME/envs/py36/bin:$MINICONDA2_HOME/envs/py36/include:$MINICONDA2_HOME/envs/py36/lib:$MINICONDA2_HOME/envs/py39/bin:$MINICONDA2_HOME/envs/py39/include:$MINICONDA2_HOME/envs/py39/lib:$PATH'
        echo ''

        echo -e 'export PATH=/usr/share/clickhouse/bin:$PATH'
        echo -e 'export PATH=/usr/share/node_exporter/bin:$PATH'
    } >>$ENV_FILE
}

#------------------------- Install the hadoop-client, hadoop-3.2.0  ------------------------
function checkHadoop() {
    local choice="$1"
    if [ -d /usr/share/"${choice}" ]; then
        echo -e "${Notice} /usr/share/${choice} is disabled, please use the /usr/share/hadoop-client"
    fi

    if [ -d /etc/"${choice}" ]; then
        echo -e "${Notice} /etc/${choice} is disabled, please use the /etc/hadoop-client"
    fi
}

function download_hadoop_client() {
    # install hadoop-client bin and external lib
    echo -e "${Info} ==> Step 1-1 -> Download the hadoop-client application package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/hadoop_package/${hadoop_client_bin}.tar.gz &&
        wget -q --no-check-certificate $FILE_SERVER/SRE/hadoop-common/${hadoop_external_lib}.tar.gz
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download hadoop package failed, please check..."
        exit 100
    fi

    # install hadoop-client conf
    echo -e "${Info} ==> Step 1-2 -> Download the hadoop configuration package .."
    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/SRE/hadoop-server-config/${hadoop_client_conf}.tar.gz
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download hadoop configuration failed, please check..."
        exit 100
    fi

    # deploy bin file and external lib file
    echo -e "${Info} ==> Step 1-3 -> Unfold the hadoop-client application and configuration package ..."
    cd $tmp_dir &&
        tar xf ${hadoop_client_bin}.tar.gz -C /opt &&
        tar xf ${hadoop_external_lib}.tar.gz -C /usr/share/java/
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold hadoop package failed, please check..."
        exit 100
    fi

    # deploy hadoop-client conf
    cd $tmp_dir &&
        mkdir -p /etc/${hadoop_client_conf} &&
        tar xf ${hadoop_client_conf}.tar.gz -C /etc/${hadoop_client_conf}
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold hadoop configuration package failed, please check ..."
        exit 100
    fi

    # create the symlink for hadoop conf
    echo -e "${Info} ==> Step 1-4 -> Create the link for hadoop-client bin and conf ..."
    removeFolder "/usr/share/hadoop-client"
    removeFolder "/usr/share/hadoop-2.10"
    removeFolder "/etc/hadoop-client"

    ln -snf /opt/${hadoop_client_bin} /usr/share/hadoop-2.10 &&
        ln -snf /usr/share/hadoop-2.10 /usr/share/hadoop-client &&
        ln -snf /etc/${hadoop_client_conf} /etc/hadoop-client
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for hadoop-client bin and conf failed, please check ..."
        exit 100
    fi
}

function download_hadoop_3_2() {
    # install hadoop 3.2 bin
    echo -e "${Info} ==> Step 1-5 -> Download the hadoop3 application package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/hadoop_package/${hadoop_3_2_bin}.tar.gz
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download hadoop3 package failed, please check..."
        exit 100
    fi

    # deploy bin file
    echo -e "${Info} ==> Step 1-6 -> Unfold the hadoop-client application and configuration package ..."
    cd $tmp_dir &&
        tar xf ${hadoop_3_2_bin}.tar.gz -C /opt
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold hadoop3 package failed, please check ..."
        exit 100
    fi

    # create the symlink for hadoop conf
    echo -e "${Info} ==> Step 1-7 -> Create the link for hadoop3 bin package ..."
    removeFolder "/usr/share/hadoop-3.2"
    ln -snf /opt/${hadoop_3_2_bin} /usr/share/hadoop-3.2
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for hadoop3 bin package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 1-8 -> Check the hadoop package deployment .."
    echo -e "${Info} ==> Step 1-8-1 -> Check the hadoop bin, /usr/share/ .."
    ls -lath /usr/share | grep hadoop
    echo -e "${Info} ==> Step 1-8-2 -> Check the hadoop conf, /etc/ .."
    ls -lath /etc | grep hadoop
    echo -e "${Info} ==> Step 1-8-3 -> Check the hadoop exterlib lib, /usr/share/java/ .."
    ls -alth /usr/share/java | grep hadoop
    echo ""
}

function install_hadoop_client() {
    echo -e "${Tip} ######## Install the hadoop-client, hadoop-3.2.0 ########"
    checkHadoop "hadoop"
    if [ -L /usr/share/hadoop-client ]; then
        echo -e "${Notice} hadoop-client is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy hadoop-client"
    fi
    download_hadoop_client
}

function install_hadoop_3_2() {
    if [ -L /usr/share/hadoop-3.2 ]; then
        echo -e "${Notice} hadoop-3.2 is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy hadoop-3.2.0"
    fi
    download_hadoop_3_2
}
#-------------------------------------------------------------------------------------------

#-------------------------- Install spark2.4, spark3.1, spark3.2 ---------------------------
function download_spark2() {
    # install spark-2.4 bin and external lib
    echo -e "${Info} ==> Step 3-1 -> Download the spark-2.4 application package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/spark_package/${spark2_bin}.tgz &&
        wget -q --no-check-certificate $FILE_SERVER/spark_package/${spark_external_lib}.tar.gz
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download spark-2.4 package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 3-2 -> Unfold the spark-2.4 application package ..."
    cd $tmp_dir &&
        tar xf ${spark2_bin}.tgz -C /opt &&
        tar xf ${spark_external_lib}.tar.gz -C /usr/share/java/
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold spark-2.4 package failed, please check ..."
        exit 100
    fi

    # create the symlink for spark bin and conf
    echo -e "${Info} ==> Step 3-3 -> Create the link for spark-2.4 bin and conf ..."
    removeFolder "/usr/share/spark"
    removeFolder "/usr/share/spark2"
    removeFolder "/usr/share/spark-2.4"
    removeFolder "/etc/spark"
    removeFolder "/etc/spark2"
    removeFolder "/etc/spark-2.4"

    rm -rf /opt/${spark2_bin}/conf &&
        ln -snf /opt/${spark2_bin} /usr/share/spark-2.4 &&
        ln -snf /usr/share/spark-2.4 /usr/share/spark2 &&
        ln -snf /usr/share/spark-2.4 /usr/share/spark &&
        mkdir -p /etc/${spark2_bin} &&
        cp -rp /opt/${spark2_bin}/conf-prod/* /etc/${spark2_bin} &&
        ln -snf /etc/${spark2_bin} /etc/spark-2.4 &&
        ln -snf /etc/spark-2.4 /etc/spark2 &&
        ln -snf /etc/spark-2.4 /etc/spark &&
        ln -snf /etc/spark-2.4 /usr/share/spark-2.4/conf
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for spark-2.4 bin and conf failed, please check ..."
        exit 100
    fi
}

function download_spark3_1() {
    # install spark-3.1 bin and external lib
    echo -e "${Info} ==> Step 3-4 -> Download the spark-3.1 application package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/spark_package/${spark3_bin}.tgz
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download spark-3.1 package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 3-5 -> Unfold the spark-3.1 application package ..."
    cd $tmp_dir &&
        tar xf ${spark3_bin}.tgz -C /opt
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold spark-3.1 package failed, please check ..."
        exit 100
    fi

    # create the symlink for spark bin and conf
    echo -e "${Info} ==> Step 3-6 -> Create the link for spark-3.1 bin and conf ..."
    removeFolder "/usr/share/spark3"
    removeFolder "/usr/share/spark-3.1"
    removeFolder "/etc/spark3"
    removeFolder "/etc/spark-3.1"

    rm -rf /opt/${spark3_bin}/conf &&
        ln -snf /opt/${spark3_bin} /usr/share/spark-3.1 &&
        ln -snf /usr/share/spark-3.1 /usr/share/spark3 &&
        mkdir -p /etc/${spark3_bin} &&
        cp -rp /opt/${spark3_bin}/conf-prod/* /etc/${spark3_bin} &&
        ln -snf /etc/${spark3_bin} /etc/spark-3.1 &&
        ln -snf /etc/spark-3.1 /etc/spark3 &&
        ln -snf /etc/spark-3.1 /usr/share/spark-3.1/conf
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for spark-3.1 bin and conf failed, please check ..."
        exit 100
    fi
}

function download_spark3_2() {
    # install spark-3.2 bin and external lib
    echo -e "${Info} ==> Step 3-7 -> Download the spark-3.2 application package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/spark_package/${spark3_2_bin}.tgz
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download spark-3.2 package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 3-8 -> Unfold the spark-3.2 application package ..."
    cd $tmp_dir &&
        tar xf ${spark3_2_bin}.tgz -C /opt
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold spark-3.2 package failed, please check ..."
        exit 100
    fi

    # create the symlink for spark bin and conf
    echo -e "${Info} ==> Step 3-9 -> Create the link for spark-3.2 bin and conf ..."
    removeFolder "/usr/share/spark-3.2"
    removeFolder "/etc/spark-3.2"

    rm -rf /opt/${spark3_2_bin}/conf &&
        ln -snf /opt/${spark3_2_bin} /usr/share/spark-3.2 &&
        mkdir -p /etc/${spark3_2_bin} &&
        cp -rp /opt/${spark3_2_bin}/conf-prod/* /etc/${spark3_2_bin} &&
        ln -snf /etc/${spark3_2_bin} /etc/spark-3.2 &&
        ln -snf /etc/spark-3.2 /usr/share/spark-3.2/conf
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for spark-3.2 bin and conf failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 3-10 -> Check the spark deployment status .."
    echo -e "${Info} ==> Step 3-10-1 -> Check the spark bin, /usr/share/ .."
    ls -alth /usr/share | grep spark
    echo -e "${Info} ==> Step 3-10-2 -> Check the spark conf, /etc/ .."
    ls -alth /etc | grep spark
    echo -e "${Info} ==> Step 3-10-3 -> Check the spark external lib, /usr/share/java/ .."
    ls -alth /usr/share/java | grep spark
    echo ""
}

function install_spark2() {
    echo -e "${Tip} ######## Install the spark-2.4, spark-3.1, spark-3.2 ########"
    if [ -L /usr/share/spark-2.4 ]; then
        echo -e "${Notice} spark2.4 is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy spark2.4"
    fi
    download_spark2
}

function install_spark3_1() {
    if [ -L /usr/share/spark-3.1 ]; then
        echo -e "${Notice} spark-3.1 is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy spark-3.1"
    fi
    download_spark3_1
}

function install_spark3_2() {
    if [ -L /usr/share/spark-3.2 ]; then
        echo -e "${Notice} spark-3.2 is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy spark3.2"
    fi
    download_spark3_2
}
#-------------------------------------------------------------------------------------------

#----------------------------------- Install Hbase -----------------------------------------
function download_hbase() {
    # install hbase bin and external lib
    echo -e "${Info} ==> Step 3-1 -> Download the hbase application package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/hbase_package/${hbase_bin}-bin.tar.gz &&
        wget -q --no-check-certificate $FILE_SERVER/hbase_package/${hbase_config} &&
        wget -q --no-check-certificate $FILE_SERVER/hbase_package/${hbase_shell_tools}
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download hbase package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 3-2 -> Unfold the hbase application package ..."
    cd $tmp_dir &&
        tar xf ${hbase_bin}-bin.tar.gz -C /opt &&
        mkdir -p /opt/hbase_tools &&
        tar xf ${hbase_shell_tools} -C /opt/hbase_tools &&
        tar xf ${hbase_config} -C /etc
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold hbase package failed, please check ..."
        exit 100
    fi

    # create the symlink for hbase bin and conf
    echo -e "${Info} ==> Step 3-3 -> Create the link for hbase bin and conf ..."
    removeFolder "/usr/share/hbase"
    removeFolder "/etc/hbase"
    removeFolder "/usr/share/hbase_tools"

    rm -rf /opt/${hbase_bin}/conf &&
        ln -snf /opt/${hbase_bin} /usr/share/hbase &&
        rm -rf /etc/${hbase_bin} &&
        mv /etc/hbase_config /etc/${hbase_bin} &&
        ln -snf /etc/${hbase_bin} /etc/hbase &&
        ln -snf /opt/hbase_tools /usr/share/hbase_tools &&
        ln -snf /etc/hbase /usr/share/hbase/conf
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for hbase bin and conf failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 3-4 -> Check the hbase install status .."
    echo -e "${Info} ==> Step 3-4-1 -> Check the hbase bin, /usr/share/ .."
    ls -alth /usr/share | grep hbase
    echo -e "${Info} ==> Step 3-4-2 -> Check the hbase conf, /etc/ .."
    ls -alth /etc/ | grep hbase
    echo ""
}

function install_hbase() {
    echo -e "${Tip} ######## Install the hbase ########"
    if [ -L /usr/share/hbase ]; then
        echo -e "${Notice} hbase is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy hbase"
    fi
    download_hbase
}

#-------------------------------------------------------------------------------------------

#----------------------------------- Install Phoenix ---------------------------------------
function download_phoenix() {
    # install phoenix bin and
    echo -e "${Info} ==> Step 4-1 -> Download the phoenix application package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/hbase_package/phoenix/${phoenix_bin}.tar.gz &&
        wget -q --no-check-certificate $FILE_SERVER/hbase_package/phoenix/${phoenix_config}.tar.gz
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download phoenix package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 4-2 -> Unfold the phoenix application package ..."
    cd $tmp_dir &&
        tar xf ${phoenix_bin}.tar.gz -C /opt &&
        tar xf ${phoenix_config}.tar.gz -C /etc
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold phoenix package failed, please check ..."
        exit 100
    fi

    # create the symlink for phoenix bin and conf
    echo -e "${Info} ==> Step 4-3 -> Create the link for phoenix bin and conf ..."
    removeFolder "/usr/share/phoenix"
    removeFolder "/etc/phoenix"
    ln -snf /opt/${phoenix_bin} /usr/share/phoenix &&
        rm -rf /etc/${phoenix_config_link} && mv /etc/${phoenix_config} /etc/${phoenix_config_link} &&
        ln -snf /etc/${phoenix_config_link} /etc/phoenix
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for phoenix bin and conf failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 4-4 -> Check the phoenix status .."
    echo -e "${Info} ==> Step 4-4-1 -> Check the phoenix bin, /usr/share/ .."
    ls -lath /usr/share | grep phoenix
    echo -e "${Info} ==> Step 4-4-2 -> Check the phoenix conf, /etc/ .."
    ls -lath /etc/ | grep phoenix
    echo ""
}

function install_phoenix() {
    echo -e "${Tip} ######## Install the phoenix ########"
    if [ -L /usr/share/phoenix ]; then
        echo -e "${Notice} phoenix is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy phoenix"
    fi
    download_phoenix
}

#-------------------------------------------------------------------------------------------

#----------------------------------- Install Clickhouse ------------------------------------
function download_clickhouse() {
    # install clickhouse bin
    echo -e "${Info} ==> Step 5-1 -> Download the clickhouse application package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/clickhouse/${clickhouse_bin}.tar.bz2
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download clickhouse package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 5-2 -> Unfold the clickhouse application package ..."
    cd $tmp_dir &&
        mkdir -p /opt/${clickhouse_bin}/bin &&
        tar xf ${clickhouse_bin}.tar.bz2 -C /opt/${clickhouse_bin}/bin/
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold clickhouse package failed, please check ..."
        exit 100
    fi

    # create the symlink for clickhouse bin and conf
    echo -e "${Info} ==> Step 5-3 -> Create the link for clickhouse bin ..."
    removeFolder "/usr/share/clickhouse"
    ln -snf /opt/${clickhouse_bin} /usr/share/clickhouse
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for clickhouse bin failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 5-3-1 -> Check the clickhouse bin, /usr/share/ .."
    ls -alth /usr/share | grep clickhouse
    echo ""
}

function install_clickhouse() {
    echo -e "${Tip} ######## Install the clickhouse ########"
    if [ -L /usr/share/clickhouse ]; then
        echo -e "${Notice} clickhouse is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy clickhouse"
    fi
    download_clickhouse
}
#-------------------------------------------------------------------------------------------

#------------------------------- Install Livy Command Tool ---------------------------------
function download_livy_command_tool() {
    # install livy command tool
    echo -e "${Info} ==> Step 6-1 -> Download the livy command tool package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/livy_package/${livy_command_tool}.tar.gz
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download livy command tool package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 6-2 -> Unfold the livy command tool package ..."
    cd $tmp_dir &&
        tar xf ${livy_command_tool}.tar.gz -C /opt/
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold livy command tool package failed, please check ..."
        exit 100
    fi

    # create the symlink for livy command tool bin and conf
    echo -e "${Info} ==> Step 6-3 -> Create the link for livy command tool bin and conf ..."
    removeFolder "/usr/share/livy"
    removeFolder "/etc/livy"

    ln -snf /opt/${livy_command_tool} /usr/share/livy &&
        rm -rf /etc/${livy_command_tool} &&
        mkdir -p /etc/${livy_command_tool} &&
        cp -rf /usr/share/livy/conf-sg/* /etc/${livy_command_tool} &&
        rm -rf /opt/${livy_command_tool}/conf &&
        ln -snf /etc/${livy_command_tool} /etc/livy &&
        ln -snf /etc/livy /usr/share/livy/conf
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create livy command tool link failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 6-3-1 -> Check the livy command tool bin, /usr/share/ .."
    ls -alth /usr/share | grep livy
    echo -e "${Info} ==> Step 6-3-2 -> Check the livy command tool conf, /etc/ .."
    ls -lath /etc/ | grep livy
    echo ""
}

function install_livy_command_tool() {
    echo -e "${Tip} ######## Install the livy command tool ########"
    if [ -L /usr/share/livy ]; then
        echo -e "${Notice} livy-command-tool is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy livy-command-tool"
    fi
    download_livy_command_tool
}
#-------------------------------------------------------------------------------------------

#----------------------------------- Install Node_exporter ---------------------------------
function download_node_exporter() {
    # install node_exporter
    echo -e "${Info} ==> Step 7-1 -> Download the node_exporter package .."
    if ! which wget >/dev/null; then
        if [ "$(check_os)" == "ubuntu" ]; then
            apt-get update >/dev/null 2>&1 && apt-get install wget -y >/dev/null 2>&1
        elif [ "$(check_os)" == "centos" ]; then
            yum update >/dev/null 2>&1 && yum install wget -y >/dev/null 2>&1
        fi
    fi

    cd $tmp_dir &&
        wget -q --no-check-certificate $FILE_SERVER/tools/${node_exporter}.tar
    if [ $? -ne 0 ]; then
        echo -e "${Error} Download node_exporter package failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 7-2 -> Unfold the node_exporter package ..."
    cd $tmp_dir &&
        rm -rf /opt/node_exporter &&
        mkdir -p /opt/node_exporter/bin &&
        tar xf ${node_exporter}.tar -C /opt/ &&
        cp -ap /opt/${node_exporter}/* /opt/node_exporter/bin/ &&
        cp -ap /opt/node_exporter/bin/node_exporter.service /etc/systemd/system/node_exporter.service
    if [ $? -ne 0 ]; then
        echo -e "${Error} Unfold node_exporter package failed, please check ..."
        exit 100
    fi

    # create the symlink for node_exporter bin and conf
    echo -e "${Info} ==> Step 7-3 -> Create the link for node_exporter bin and conf ..."
    removeFolder "/usr/share/node_exporter"
    ln -snf /opt/${node_exporter} /usr/share/node_exporter
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for node_exporter bin failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 7-3-1 -> Check the node_exporter bin, /usr/share/ .."
    ls -alth /usr/share | grep node_exporter
}

function start_node_exporter() {
    echo -e "${Info} ==> Step 7-4 -> Start the node_exporter service .."

    N=$(netstat -antplue | egrep -i listen | egrep ":9100 " | wc -l)
    if [ $N -gt 0 ]; then
        echo -e "${Error} Sorry . I cant install node_exporter cause port 9100 is in used by another process. [You can disable another node_exporter and install me again]"
        echo ""
    else
        echo -e "${Info} Enable node_exporter auto start and make it a system service and start node_exporter"
        systemctl enable node_exporter.service
        systemctl start node_exporter.service
        sleep 2
        N=$(netstat -antplue | egrep -i listen | egrep ":9100 " | wc -l)
        if [ $N -eq 1 ]; then
            echo -e "${Info} start node_exporter Successfully"
            echo ""
        else
            echo -e "${Error} start node_exporter Failed!!!"
            echo ""
        fi
    fi
}

function install_node_exporter() {
    echo -e "${Tip} ######## Install the node_exporter ########"
    if [ -L /usr/share/node_exporter ]; then
        echo -e "${Notice} node_exporter is already installed, will upgrade the version!!!"
    else
        echo -e "${Info} deploy node_exporter"
    fi
    download_node_exporter
    start_node_exporter
}

#-------------------------------------------------------------------------------------------

#----------------------------------- Install Miniconda2 ---------------------------------
function install_miniconda2() {
    echo -e "${Tip} ######## Install the miniconda2 ########"
    echo -e "${Info} ==> Step 8-1 -> Check the miniconda2 deployment .."
    if [ -d /opt/miniconda2 ]; then
        echo -e "${Info} ==> Step 8-2 -> Miniconda2 folder has been in /opt/miniconda2, will check the /usr/share/miniconda2"
    else
        echo -e "${Info} ==> Step 8-2 -> Download Miniconda2 package ..."

        cd $tmp_dir &&
            wget -q --no-check-certificate $FILE_SERVER/tools/${miniconda2} &&
            tar xf ${miniconda2} -C /opt/
        if [ $? -ne 0 ]; then
            echo -e "${Error} Download or Unfold the Miniconda2 package failed, please check ..."
            exit 100
        fi
    fi

    removeFolder "/usr/share/miniconda2"
    echo -e "${Info} ==> Step 8-3 -> Create the link for miniconda2 bin ..."
    ln -snf /opt/miniconda2 /usr/share/miniconda2
    if [ $? -ne 0 ]; then
        echo -e "${Error} Create the link for miniconda2 bin failed, please check ..."
        exit 100
    fi

    echo -e "${Info} ==> Step 8-4 -> Check the miniconda2 bin, /usr/share/ .."
    ls -alth /usr/share | grep miniconda2
    echo ""

}
#-------------------------------------------------------------------------------------------

function collect_driver_info() {
    if [ ! -f "$version_file" ]; then
        mkdir -p /opt/di-driver-update/
        echo "$version" >"$version_file"
        echo -e "${Green_font_prefix}We need to collect some info about this first time, please provide it!!!${Font_color_suffix}"
        read -p "Enter your machine PIC, eamil prefix: " username
        read -p "Enter your projectCode in RAM: " projectCode
        read -p "Enter your teamCode in RAM: " teamCode
        ip=$(hostname -i)
        hostname=$(hostname)

        while [ "$username" == "" ] || [ "$projectCode" == "" ] || [ "$teamCode" == "" ]; do
            echo ""
            echo -e "${Yellow_font_prefix}[Username/ProjectCode/TeamCode NULL, please input it.]${Font_color_suffix}"
            read -p "Enter the machine PIC, email prefix: " username
            read -p "Enter your projectCode in RAM: " projectCode
            read -p "Enter your teamCode in RAM: " teamCode
        done

        curl -s --request POST '10.128.143.230:9000/api/v1/privateDriver' \
            --header 'Content-Type: application/json' \
            -d '{
                "ip": "'$ip'",
                "hostname": "'$hostname'",
                "username": "'$username'",
                "teamCode": "'$teamCode'",
                "projectCode": "'$projectCode'",
                "version": "'$version'"
            }' >/dev/null

        if [ $? -ne 0 ]; then
            echo -e "${Yellow_font_prefix}[collect info failed, please contact zhipeng.wangwzp@shopee.com]${Font_color_suffix}"
        fi

        echo "ip: $ip" >>/opt/di-driver-update/info
        echo "hostname: $hostname" >>/opt/di-driver-update/info
        echo "PIC: $username" >>/opt/di-driver-update/info
        echo "ProjectCode: $projectCode" >>/opt/di-driver-update/info
        echo "TeamCode: $teamCode" >>/opt/di-driver-update/info

    fi
}

function removeUpdateMotd() {
    sed -i '/upgrade\|di-driver-update/d' /etc/motd
}

#-------------------------- 3. User Input ------------------------
function userInput() {
    echo -e && echo -e " Shopee DataInfra Driver related component deploy script
  -- any concern please contact zhipeng.wangwzp@shopee.com  --

 ${Green_font_prefix}0.${Font_color_suffix} install all [hadoop,spark,hbase,phoenix,streamee,clickhouse,livy]
 ${Green_font_prefix}1.${Font_color_suffix} install hadoop
 ${Green_font_prefix}2.${Font_color_suffix} install spark [need install hadoop]
 ${Green_font_prefix}3.${Font_color_suffix} install hbase
 ${Green_font_prefix}4.${Font_color_suffix} install phoenix
 ${Green_font_prefix}5.${Font_color_suffix} install clickhouse
 ${Green_font_prefix}6.${Font_color_suffix} install livy-command-tool
 ${Green_font_prefix}7.${Font_color_suffix} install node_exporter
 ${Green_font_prefix}8.${Font_color_suffix} install miniconda2
 ————————————" && echo
    read -r -e -p " Input your choice [0-8]: " num
    case "$num" in
    0)
        echo -e "${Tip} Install the hadoop-client, hadoop-3.2.0, spark-2.4, spark-3.1, spark-3.2, hbase, phoenix, clickhouse, livy, node_exporter, miniconda2"
        check_jdk
        install_hadoop_client
        install_hadoop_3_2
        install_spark2
        install_spark3_1
        install_spark3_2
        install_hbase
        install_phoenix
        install_streamee
        install_clickhouse
        install_livy_command_tool
        install_node_exporter
        install_miniconda2
        add_to_motd
        setup_env
        removeUpdateMotd
        ;;
    1)
        check_jdk
        install_hadoop_client
        install_hadoop_3_2
        add_to_motd
        setup_env
        removeUpdateMotd
        ;;
    2)
        check_jdk
        install_spark2
        install_spark3_1
        install_spark3_2
        add_to_motd
        setup_env
        removeUpdateMotd
        ;;
    3)
        check_jdk
        install_hbase
        add_to_motd
        setup_env
        removeUpdateMotd
        ;;
    4)
        check_jdk
        install_phoenix
        add_to_motd
        setup_env
        removeUpdateMotd
        ;;
    5)
        check_jdk
        install_clickhouse
        add_to_motd
        setup_env
        removeUpdateMotd
        ;;
    6)
        check_jdk
        install_livy_command_tool
        add_to_motd
        setup_env
        removeUpdateMotd
        ;;
    7)
        check_jdk
        install_node_exporter
        setup_env
        ;;
    8)
        check_jdk
        install_miniconda2
        setup_env
        ;;
    *)
        echo -e "Please input the right num [0-8]"
        ;;
    esac
}

# -------------------------- 4. Main --------------------------
checkRoot

# mk tmp, java, sparklocaldir
mkTmp

# set all variables
setVariable

# collect info
collect_driver_info

# user input
userInput
