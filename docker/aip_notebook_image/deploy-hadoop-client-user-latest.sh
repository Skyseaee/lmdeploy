USER_TYPE=user
BINARY_210_VERSION=hadoop-2.10.sdi-080
BINARY_32_VERSION=hadoop-3.2.sdi-001
BINARY_33_VERSION=hadoop-3.3.sdi-069
CONF_VERSION_BATCH_SG=hadoop-client-Prod-TL0-sdi-189
CONF_VERSION_BATCH_US=hadoop-client-Prod-ASB-sdi-047
REGION=$1


# deploy hadoop client binary
deploy_bin () {
	for BINARY_VERSION in $(if [ ${USER_TYPE} == "user" ]; then echo ${BINARY_210_VERSION}; elif [ ${USER_TYPE} == "livy" ]; then echo ${BINARY_210_VERSION} ${BINARY_32_VERSION} ${BINARY_33_VERSION}; else exit 1; fi);  
    do 
        for CLUSTER_TYPE in $(if [ ${USER_TYPE} == "user" ]; then echo batch; elif [ ${USER_TYPE} == "livy" ]; then echo batch streaming; else exit 1; fi);  
        do 
            BIN_DIR=${BINARY_VERSION}-client-${CLUSTER_TYPE} \
            && BIN_DIR_WORK=hadoop-$(echo ${BINARY_VERSION} | awk -F '.sdi' '{print $1}' | awk -F '-' '{print $2}')-client-${CLUSTER_TYPE} \
            && CONF_DIR_WORK=/etc/${BIN_DIR_WORK} \
            && cd /opt \
            && wget http://nexus-repo.data-infra.shopee.io/repository/file_server/hadoop_package/${BINARY_VERSION}.tar.gz \
            && tar zxf ${BINARY_VERSION}.tar.gz \
            && mv ${BINARY_VERSION} ${BIN_DIR} \
            && ln -s /opt/${BIN_DIR} /usr/share/${BIN_DIR_WORK} \
            && rm -rf /usr/share/${BIN_DIR_WORK}/etc/hadoop \
            && ln -s ${CONF_DIR_WORK} /usr/share/${BIN_DIR_WORK}/etc/hadoop \
            && ln -s /usr/share/${BIN_DIR_WORK} /usr/share/hadoop-client \
            && rm -f ${BINARY_VERSION}.tar.gz; 
        done; 
    done 
}


# deploy hadoop client configuration 
deploy_conf () {
    for BINARY_VERSION in $(if [ ${USER_TYPE} == "user" ]; then echo ${BINARY_210_VERSION}; elif [ ${USER_TYPE} == "livy" ]; then echo ${BINARY_210_VERSION} ${BINARY_32_VERSION} ${BINARY_33_VERSION}; else exit 1; fi);  
    do 
        for CONF_VERSION in $(if [ ${REGION} == "sg" ]; then echo ${CONF_VERSION_BATCH_SG}; elif [ ${REGION} == "us" ]; then echo ${CONF_VERSION_BATCH_US}; else echo REGION: ${REGION}, should be 'sg' or 'us'; exit 1; fi); 
        do 
            CLUSTER_TYPE="batch" \
            && CONF_DIR=hadoop-$(echo $BINARY_VERSION | awk -F '.sdi' '{print $1}' | awk -F '-' '{print $2}')-client-$(echo ${CONF_VERSION} | awk -F 'hadoop-client-' '{print $2}') \
            && CONF_DIR_WORK=hadoop-$(echo $BINARY_VERSION | awk -F '.sdi' '{print $1}' | awk -F '-' '{print $2}')-client-${CLUSTER_TYPE} \
            && cd /etc \
            && wget http://nexus-repo.data-infra.shopee.io/repository/file_server/SRE/hadoop-server-config/${CONF_VERSION}.tar.gz \
            && mkdir ${CONF_DIR} \
            && tar zxf ${CONF_VERSION}.tar.gz -C ${CONF_DIR} \
            && ln -s /etc/${CONF_DIR} /etc/${CONF_DIR_WORK} \
            && rm -f ${CONF_VERSION}.tar.gz; \
        done; 
    done
}

#ENV PATH=$PATH:$HOME/bin
#ENV HADOOP_HOME=/usr/share/hadoop-client
#ENV PATH=$HADOOP_HOME/bin:$HADOOP_HOME/include:$HADOOP_HOME/lib:$HADOOP_HOME/libexec:$HADOOP_HOME/sbin:$HADOOP_HOME/share:$PATH
#
deploy_bin  
deploy_conf
